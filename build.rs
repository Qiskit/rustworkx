use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::io::Read;
use std::io::Write;
use std::path;

fn main() {
    let src_dir_path = env::var("CARGO_MANIFEST_DIR").unwrap();
    let src_dir_path = format!("{}/src/", src_dir_path);
    let out_dir = fs::read_dir(src_dir_path).expect("could not read src/ directory");

    let mut modules = BTreeSet::new();
    for entry in out_dir {
        let entry = entry.expect("could not read entry");
        let path = entry.path();
        // Top-level .rs files
        if path.is_file() {
            let file_name = path.to_str().unwrap();
            if file_name.ends_with(".rs") && file_name != "lib.rs" {
                modules.insert(file_name.to_string());
            }
        }
        if path.is_dir() {
            let sub_dir = fs::read_dir(path).expect("could not read subdirectory");
            for entry in sub_dir {
                let entry = entry.expect("could not read entry");
                let path = entry.path();
                if path.is_file() {
                    let file_name = path.to_str().unwrap();
                    if file_name.ends_with("mod.rs") {
                        let module_name = file_name.to_string();
                        modules.insert(module_name);
                    }
                }
            }
        }
    }

    // Create the generated file with the modules
    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = path::PathBuf::from(out_dir).join("generated_include_rustworkx_modules.rs");

    // Create the file and write the contents to it
    let mut f = fs::File::create(dest_path).unwrap();

    let mut rustworkx_modules = BTreeSet::new();

    for path in modules {
        let mut file = fs::File::open(path.clone())
            .expect("could not open file to check if it declares a rustworkx module");
        let mut content = String::new();
        file.read_to_string(&mut content)
            .expect("could not read contents of the file");
        if content.contains("declare_rustworkx_module!") {
            rustworkx_modules.insert(module_name_from_file_name(path.clone()));
        }
    }

    writeln!(f, "fn register_rustworkx_modules(m: &pyo3::Bound<pyo3::types::PyModule>) ->  pyo3::prelude::PyResult<()>").expect("could not write function signature");
    writeln!(f, "{{").expect("could not write function body");

    for module in rustworkx_modules {
        writeln!(f, "{}::rustworkx_module(m)?;", module.clone()).expect("could not write to file");
    }
    writeln!(f, "Ok(())").expect("could not write function body");
    writeln!(f, "}}").expect("could not write function body");
}

fn module_name_from_file_name(filename: String) -> String {
    if filename.ends_with("mod.rs") {
        let parent = path::Path::new(&filename)
            .parent()
            .expect("could not get parent directory");
        let module_name = parent
            .file_name()
            .expect("could not get file name")
            .to_str()
            .expect("could not convert to string");
        return module_name.to_string();
    }
    let module_name = path::Path::new(&filename)
        .file_name()
        .expect("could not get file name")
        .to_str()
        .expect("could not convert to string");
    return module_name
        .to_string()
        .strip_suffix(".rs")
        .expect("could not strip suffix")
        .to_string();
}
