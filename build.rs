#[cfg(windows)]
fn main() {
    let sdk_bin = r"C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64";
    if let Ok(current_path) = std::env::var("PATH") {
        std::env::set_var("PATH", format!("{};{}", sdk_bin, current_path));
    }

    let mut res = winres::WindowsResource::new();
    res.set("CompanyName", "Pyintel Core Intelligence");
    res.set("FileDescription", "Pyintel Mem (.pmem) Engine");
    res.set("ProductName", "Pyintel Mem");
    res.set("LegalCopyright", "Copyright © 2026 Pyintel. All rights reserved.");
    res.set("OriginalFilename", "pmem.exe");
    res.set("InternalName", "pmem");
    res.set_version_info(winres::VersionInfo::PRODUCTVERSION, 0x0001000000000000);
    res.set_version_info(winres::VersionInfo::FILEVERSION, 0x0001000000000000);
    if let Err(e) = res.compile() {
        eprintln!("Warning: failed to compile Windows resources: {}", e);
    }
}

#[cfg(not(windows))]
fn main() {}
