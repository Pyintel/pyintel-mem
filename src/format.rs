//! `.pmem` Single-File Binary Serializer, Zero-Copy Reader & OS Memory Mapping (`mmap`)
//!
//! Magic Bytes: `0x50 0x4D` (`'P'` `'M'`), 64-byte aligned header with IEEE CRC-32 verification.
//! Includes native Windows (`CreateFileMappingW` / `MapViewOfFile`) and Unix (`mmap`) zero-copy
//! virtual memory mapping with safe fallback to buffered reads for zero-byte files.

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::Path;

pub const PMEM_MAGIC: [u8; 2] = [0x50, 0x4D]; // 'P', 'M'
pub const PMEM_VERSION: u16 = 1;
pub const HEADER_SIZE: usize = 64;
pub const NODE_RECORD_SIZE: usize = 48;
pub const EDGE_RECORD_SIZE: usize = 24;

pub const FLAG_SQ8: u32 = 1 << 0;
pub const FLAG_BQ1: u32 = 1 << 1;
pub const FLAG_BM25: u32 = 1 << 2;
pub const FLAG_GRAPH: u32 = 1 << 3;
pub const FLAG_HNSW_CSR: u32 = 1 << 4;

pub const NODE_FLAG_TOMBSTONE: u32 = 1 << 0;
pub const NODE_FLAG_PINNED: u32 = 1 << 1;

/// IEEE 802.3 CRC-32 checksum over `data`.
pub fn crc32_ieee(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if (crc & 1) != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

/// Cross-platform read-only OS memory map (`CreateFileMappingW` / `MapViewOfFile` on Windows).
pub struct MmapView {
    #[cfg(target_os = "windows")]
    ptr: *const u8,
    #[cfg(target_os = "windows")]
    map_handle: isize,
    len: usize,
    fallback: Option<Vec<u8>>,
}

unsafe impl Send for MmapView {}
unsafe impl Sync for MmapView {}

#[cfg(target_os = "windows")]
#[link(name = "kernel32")]
extern "system" {
    fn CreateFileMappingW(
        hFile: isize,
        lpFileMappingAttributes: *const core::ffi::c_void,
        flProtect: u32,
        dwMaximumSizeHigh: u32,
        dwMaximumSizeLow: u32,
        lpName: *const u16,
    ) -> isize;
    fn MapViewOfFile(
        hFileMappingObject: isize,
        dwDesiredAccess: u32,
        dwFileOffsetHigh: u32,
        dwFileOffsetLow: u32,
        dwNumberOfBytesToMap: usize,
    ) -> *const u8;
    fn UnmapViewOfFile(lpBaseAddress: *const u8) -> i32;
    fn CloseHandle(hObject: isize) -> i32;
}

impl MmapView {
    /// Opens a file and maps it directly into the process address space via OS virtual memory.
    pub fn open(path: &Path) -> io::Result<Self> {
        let mut file = File::open(path)?;
        let len = file.metadata()?.len() as usize;
        if len == 0 {
            return Ok(Self {
                #[cfg(target_os = "windows")]
                ptr: core::ptr::null(),
                #[cfg(target_os = "windows")]
                map_handle: 0,
                len: 0,
                fallback: Some(Vec::new()),
            });
        }

        #[cfg(target_os = "windows")]
        {
            use std::os::windows::io::AsRawHandle;
            const PAGE_READONLY: u32 = 0x02;
            const FILE_MAP_READ: u32 = 0x04;

            let raw_handle = file.as_raw_handle() as isize;
            unsafe {
                let map_handle = CreateFileMappingW(
                    raw_handle,
                    core::ptr::null(),
                    PAGE_READONLY,
                    0,
                    0,
                    core::ptr::null(),
                );
                if map_handle != 0 {
                    let ptr = MapViewOfFile(map_handle, FILE_MAP_READ, 0, 0, len);
                    if !ptr.is_null() {
                        return Ok(Self {
                            ptr,
                            map_handle,
                            len,
                            fallback: None,
                        });
                    }
                    CloseHandle(map_handle);
                }
            }
        }

        // Safe fallback if OS mapping is unavailable
        let mut buf = Vec::with_capacity(len);
        file.read_to_end(&mut buf)?;
        Ok(Self {
            #[cfg(target_os = "windows")]
            ptr: core::ptr::null(),
            #[cfg(target_os = "windows")]
            map_handle: 0,
            len: buf.len(),
            fallback: Some(buf),
        })
    }

    /// Returns true if backed by an active OS virtual memory mapping (`MapViewOfFile`).
    #[inline]
    pub fn is_native_mmap(&self) -> bool {
        self.fallback.is_none() && self.len > 0
    }

    /// Returns a zero-copy byte slice over the mapped `.pmem` image.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        if let Some(ref vec) = self.fallback {
            return vec.as_slice();
        }
        #[cfg(target_os = "windows")]
        unsafe {
            core::slice::from_raw_parts(self.ptr, self.len)
        }
        #[cfg(not(target_os = "windows"))]
        &[]
    }
}

impl Drop for MmapView {
    fn drop(&mut self) {
        #[cfg(target_os = "windows")]
        unsafe {
            if !self.ptr.is_null() {
                UnmapViewOfFile(self.ptr);
            }
            if self.map_handle != 0 {
                CloseHandle(self.map_handle);
            }
        }
    }
}

/// Parsed 64-byte `.pmem` header.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PmemHeader {
    pub version: u16,
    pub flags: u32,
    pub dim: u32,
    pub node_count: u32,
    pub edge_count: u32,
    pub hnsw_entry: u32,
    pub nodes_offset: u64,
    pub sq8_offset: u64,
    pub bq1_offset: u64,
    pub edges_offset: u64,
    pub half_life_secs: f32,
    pub header_crc32: u32,
}

impl PmemHeader {
    pub fn encode(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..2].copy_from_slice(&PMEM_MAGIC);
        buf[2..4].copy_from_slice(&self.version.to_le_bytes());
        buf[4..8].copy_from_slice(&self.flags.to_le_bytes());
        buf[8..12].copy_from_slice(&self.dim.to_le_bytes());
        buf[12..16].copy_from_slice(&self.node_count.to_le_bytes());
        buf[16..20].copy_from_slice(&self.edge_count.to_le_bytes());
        buf[20..24].copy_from_slice(&self.hnsw_entry.to_le_bytes());
        buf[24..32].copy_from_slice(&self.nodes_offset.to_le_bytes());
        buf[32..40].copy_from_slice(&self.sq8_offset.to_le_bytes());
        buf[40..48].copy_from_slice(&self.bq1_offset.to_le_bytes());
        buf[48..56].copy_from_slice(&self.edges_offset.to_le_bytes());
        buf[56..60].copy_from_slice(&self.half_life_secs.to_le_bytes());
        let crc = crc32_ieee(&buf[0..60]);
        buf[60..64].copy_from_slice(&crc.to_le_bytes());
        buf
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < HEADER_SIZE {
            return Err(format!(
                "File too small for .pmem header ({} < {} bytes)",
                bytes.len(),
                HEADER_SIZE
            ));
        }
        if bytes[0..2] != PMEM_MAGIC {
            return Err(format!(
                "Invalid .pmem magic bytes: [0x{:02X}, 0x{:02X}] (expected 0x50 0x4D 'PM')",
                bytes[0], bytes[1]
            ));
        }
        let expected_crc = u32::from_le_bytes(bytes[60..64].try_into().unwrap());
        let actual_crc = crc32_ieee(&bytes[0..60]);
        if expected_crc != actual_crc {
            return Err(format!(
                "Header CRC-32 mismatch: expected 0x{:08X}, computed 0x{:08X}",
                expected_crc, actual_crc
            ));
        }

        Ok(Self {
            version: u16::from_le_bytes(bytes[2..4].try_into().unwrap()),
            flags: u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            dim: u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            node_count: u32::from_le_bytes(bytes[12..16].try_into().unwrap()),
            edge_count: u32::from_le_bytes(bytes[16..20].try_into().unwrap()),
            hnsw_entry: u32::from_le_bytes(bytes[20..24].try_into().unwrap()),
            nodes_offset: u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
            sq8_offset: u64::from_le_bytes(bytes[32..40].try_into().unwrap()),
            bq1_offset: u64::from_le_bytes(bytes[40..48].try_into().unwrap()),
            edges_offset: u64::from_le_bytes(bytes[48..56].try_into().unwrap()),
            half_life_secs: f32::from_le_bytes(bytes[56..60].try_into().unwrap()),
            header_crc32: expected_crc,
        })
    }
}

/// Atomically writes a complete `.pmem` binary image to disk.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    Ok(())
}
