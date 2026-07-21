//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/CommandLine/CommandLine.hpp>
#include <Einsums/Config/Types.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/Tensor/InitModule.hpp>
#include <Einsums/Tensor/ModuleVars.hpp>

#include <H5Fpublic.h>
#include <H5Ppublic.h>
#include <H5Tdevelop.h>
#include <H5public.h>
#include <filesystem>
#include <mutex>
#include <string>

namespace einsums {

/*
 * Set up the internal state of the module. If the module does not need to be set up, then this
 * file can be safely deleted. Make sure that if you do, you also remove its reference in the CMakeLists.txt,
 * as well as the initialization header for the module and the dependence on Einsums_Runtime, assuming these
 * aren't being used otherwise.
 *
 * Logging will not be available by the time the initialization routines are run.
 */

int setup_Einsums_Tensor() {
    // Auto-generated code. Do not touch if you are unsure of what you are doing.
    // Instead, modify the other functions below.
    // If you don't need a function, you may remove its respective line from the
    // if statement below.
    static bool is_initialized = false;

    if (!is_initialized) {
        einsums::register_arguments(einsums::add_Einsums_Tensor_arguments);
        einsums::register_startup_function(einsums::initialize_Einsums_Tensor);
        einsums::register_shutdown_function(einsums::finalize_Einsums_Tensor);

        is_initialized = true;
    }

    return 0;
}

EINSUMS_EXPORT void add_Einsums_Tensor_arguments() {

    auto pid = getpid();

    auto &global_config = GlobalConfigMap::get_singleton();
    auto &global_string = global_config.get_string_map()->get_value();
    auto &global_double = global_config.get_double_map()->get_value();
    auto &global_int    = global_config.get_int_map()->get_value();
    auto &global_bool   = global_config.get_bool_map()->get_value();

    auto lock = std::lock_guard(global_config);

    static cl::OptionCategory TensorCategory("Tensor Options");

    static cl::Opt<std::string> scratch_dir("einsums:scratch-dir", {}, "The scratch directory for Einsums tensor files.", TensorCategory,
                                            cl::Location(global_string["scratch-dir"]),
                                            cl::Default(std::filesystem::temp_directory_path().string()));
    static cl::Opt<std::string> file_name(
        "einsums:hdf5-file-name", {},
        "The name of the HDF5 file for Einsums. Defaults to einsums.[pid].h5, where [pid] is the PID of the current process.",
        TensorCategory, cl::Location(global_string["hdf5-file-name"]), cl::Default(fmt::format("einsums.{}.h5", pid)));

    // delete-hdf5-files drives cleanup in finalize_Einsums_Tensor. It must default
    // to true so the per-process scratch file (einsums.<pid>.h5) is removed on exit;
    // otherwise these files accumulate indefinitely in the scratch dir and, via PID
    // reuse, a later process inherits a stale one. Passing --einsums:no-delete-hdf5-files
    // sets it false to keep the files, matching the flag's name.
    static cl::Flag delete_files("einsums:no-delete-hdf5-files", {}, "Tells Einsums not to clean up HDF5 files on exit.", TensorCategory,
                                 cl::Location(global_bool["delete-hdf5-files"]), cl::Default(true), cl::ImplicitValue(false));
}

static void create_complex_types() {
    auto &singleton     = einsums::detail::Einsums_Tensor_vars::get_singleton();
    auto &global_config = GlobalConfigMap::get_singleton();

    singleton.double_complex_type = H5Tcreate(H5T_COMPOUND, 2 * sizeof(double));
    singleton.float_complex_type  = H5Tcreate(H5T_COMPOUND, 2 * sizeof(float));

    if (singleton.double_complex_type == H5I_INVALID_HID) {
        EINSUMS_LOG_ERROR("Could not create HDF5 double complex number data type!");
        H5Fclose(singleton.hdf5_file);
        H5Pclose(singleton.link_property_list);
        std::terminate();
    } else {
        int err = 1;
        err     = H5Tinsert(singleton.double_complex_type, "x", 0, H5T_NATIVE_DOUBLE);

        if (err < 0) {
            EINSUMS_LOG_ERROR("Could not assign members to double complex data type!");
            H5Fclose(singleton.hdf5_file);
            H5Pclose(singleton.link_property_list);
            H5Fclose(singleton.double_complex_type);
            std::terminate();
        }

        err = H5Tinsert(singleton.double_complex_type, "y", 8, H5T_NATIVE_DOUBLE);

        if (err < 0) {
            EINSUMS_LOG_ERROR("Could not assign members to double complex data type!");
            H5Fclose(singleton.hdf5_file);
            H5Pclose(singleton.link_property_list);
            H5Fclose(singleton.double_complex_type);
            std::terminate();
        }

        err = H5Tcommit(singleton.hdf5_file, "double-complex", singleton.double_complex_type, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        if (err < 0) {
            EINSUMS_LOG_ERROR("Could not commit double complex data type!");
            H5Fclose(singleton.hdf5_file);
            H5Pclose(singleton.link_property_list);
            H5Fclose(singleton.double_complex_type);
            std::terminate();
        }
    }

    if (singleton.float_complex_type == H5I_INVALID_HID) {
        EINSUMS_LOG_ERROR("Could not create HDF5 float complex number data type!");
        H5Fclose(singleton.hdf5_file);
        H5Pclose(singleton.link_property_list);
        H5Fclose(singleton.double_complex_type);
        std::terminate();
    } else {
        int err = 1;
        err     = H5Tinsert(singleton.float_complex_type, "x", 0, H5T_NATIVE_FLOAT);

        if (err < 0) {
            EINSUMS_LOG_ERROR("Could not assign members to float complex data type!");
            H5Fclose(singleton.hdf5_file);
            H5Pclose(singleton.link_property_list);
            H5Fclose(singleton.double_complex_type);
            H5Fclose(singleton.float_complex_type);
            std::terminate();
        }

        err = H5Tinsert(singleton.float_complex_type, "y", 4, H5T_NATIVE_FLOAT);

        if (err < 0) {
            EINSUMS_LOG_ERROR("Could not assign members to float complex data type!");
            H5Fclose(singleton.hdf5_file);
            H5Pclose(singleton.link_property_list);
            H5Fclose(singleton.double_complex_type);
            H5Fclose(singleton.float_complex_type);
            std::terminate();
        }

        err = H5Tcommit(singleton.hdf5_file, "float-complex", singleton.float_complex_type, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        if (err < 0) {
            EINSUMS_LOG_ERROR("Could not commit float complex data type!");
            H5Fclose(singleton.hdf5_file);
            H5Pclose(singleton.link_property_list);
            H5Fclose(singleton.double_complex_type);
            H5Fclose(singleton.float_complex_type);
            std::terminate();
        }
    }
}

static void open_complex_types() {
    auto &singleton = einsums::detail::Einsums_Tensor_vars::get_singleton();

    singleton.double_complex_type = H5Topen(singleton.hdf5_file, "double-complex", H5P_DEFAULT);
    singleton.float_complex_type  = H5Topen(singleton.hdf5_file, "float-complex", H5P_DEFAULT);

    if (singleton.double_complex_type == H5I_INVALID_HID) {
        singleton.double_complex_type = H5Tcreate(H5T_COMPOUND, 2 * sizeof(double));

        if (singleton.double_complex_type == H5I_INVALID_HID) {
            EINSUMS_LOG_ERROR("Could not create HDF5 double complex number data type!");
            H5Fclose(singleton.hdf5_file);
            H5Pclose(singleton.link_property_list);
            std::terminate();
        } else {
            int err = 1;
            err     = H5Tinsert(singleton.double_complex_type, "x", 0, H5T_NATIVE_DOUBLE);

            if (err < 0) {
                EINSUMS_LOG_ERROR("Could not assign members to double complex data type!");
                H5Fclose(singleton.hdf5_file);
                H5Pclose(singleton.link_property_list);
                H5Fclose(singleton.double_complex_type);
                std::terminate();
            }

            err = H5Tinsert(singleton.double_complex_type, "y", 8, H5T_NATIVE_DOUBLE);

            if (err < 0) {
                EINSUMS_LOG_ERROR("Could not assign members to double complex data type!");
                H5Fclose(singleton.hdf5_file);
                H5Pclose(singleton.link_property_list);
                H5Fclose(singleton.double_complex_type);
                std::terminate();
            }

            err = H5Tcommit(singleton.hdf5_file, "double-complex", singleton.double_complex_type, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

            if (err < 0) {
                EINSUMS_LOG_ERROR("Could not commit double complex data type!");
                H5Fclose(singleton.hdf5_file);
                H5Pclose(singleton.link_property_list);
                H5Fclose(singleton.double_complex_type);
                std::terminate();
            }
        }
    }

    if (singleton.float_complex_type == H5I_INVALID_HID) {
        singleton.float_complex_type = H5Tcreate(H5T_COMPOUND, 2 * sizeof(float));

        if (singleton.float_complex_type == H5I_INVALID_HID) {
            EINSUMS_LOG_ERROR("Could not create HDF5 float complex number data type!");
            H5Fclose(singleton.hdf5_file);
            H5Pclose(singleton.link_property_list);
            H5Fclose(singleton.double_complex_type);
            std::terminate();
        } else {
            int err = 1;
            err     = H5Tinsert(singleton.float_complex_type, "x", 0, H5T_NATIVE_FLOAT);

            if (err < 0) {
                EINSUMS_LOG_ERROR("Could not assign members to float complex data type!");
                H5Fclose(singleton.hdf5_file);
                H5Pclose(singleton.link_property_list);
                H5Fclose(singleton.double_complex_type);
                H5Fclose(singleton.float_complex_type);
                std::terminate();
            }

            err = H5Tinsert(singleton.float_complex_type, "y", 4, H5T_NATIVE_FLOAT);

            if (err < 0) {
                EINSUMS_LOG_ERROR("Could not assign members to float complex data type!");
                H5Fclose(singleton.hdf5_file);
                H5Pclose(singleton.link_property_list);
                H5Fclose(singleton.double_complex_type);
                H5Fclose(singleton.float_complex_type);
                std::terminate();
            }

            err = H5Tcommit(singleton.hdf5_file, "float-complex", singleton.float_complex_type, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

            if (err < 0) {
                EINSUMS_LOG_ERROR("Could not commit float complex data type!");
                H5Fclose(singleton.hdf5_file);
                H5Pclose(singleton.link_property_list);
                H5Fclose(singleton.double_complex_type);
                H5Fclose(singleton.float_complex_type);
                std::terminate();
            }
        }
    }
}

bool open_hdf5_file(std::string const &fname) {
    auto &singleton     = einsums::detail::Einsums_Tensor_vars::get_singleton();
    auto &global_config = GlobalConfigMap::get_singleton();

    singleton.hdf5_file = H5Fopen(fname.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);

    if (singleton.hdf5_file == H5I_INVALID_HID) {
        // The file exists but is not a usable HDF5 file. This is not fatal: the
        // scratch file is named einsums.<pid>.h5 and is not deleted by default,
        // so a corrupt or truncated leftover from an earlier process whose PID
        // has been reused would land here. Signal the caller to recreate it
        // rather than terminating the whole process.
        EINSUMS_LOG_WARN("Existing HDF5 file '{}' could not be opened (stale or corrupt); it will be recreated.", fname);
        return false;
    }

    singleton.link_property_list = H5Pcreate(H5P_LINK_CREATE);

    if (singleton.link_property_list == H5I_INVALID_HID) {
        EINSUMS_LOG_ERROR("Could not create HDF5 link property list!");
        H5Fclose(singleton.hdf5_file);
        std::terminate();
    }

    int res = H5Pset_char_encoding(singleton.link_property_list, H5T_CSET_UTF8);

    if (res < 0) {
        EINSUMS_LOG_ERROR("Could not apply properties to the HDF5 link property list!");
        H5Fclose(singleton.hdf5_file);
        H5Pclose(singleton.link_property_list);
        std::terminate();
    }

    res = H5Pset_create_intermediate_group(singleton.link_property_list, 1);

    if (res < 0) {
        EINSUMS_LOG_ERROR("Could not apply properties to the HDF5 link property list!");
        H5Fclose(singleton.hdf5_file);
        H5Pclose(singleton.link_property_list);
        std::terminate();
    }

    open_complex_types();
    return true;
}

void create_hdf5_file(std::string const &fname) {
    auto &singleton     = einsums::detail::Einsums_Tensor_vars::get_singleton();
    auto &global_config = GlobalConfigMap::get_singleton();

    singleton.hdf5_file = H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    if (singleton.hdf5_file == H5I_INVALID_HID) {
        EINSUMS_LOG_ERROR("HDF5 file could not be opened!");
        std::terminate();
    }

    singleton.link_property_list = H5Pcreate(H5P_LINK_CREATE);

    if (singleton.link_property_list == H5I_INVALID_HID) {
        EINSUMS_LOG_ERROR("Could not create HDF5 link property list!");
        H5Fclose(singleton.hdf5_file);
        std::terminate();
    }

    int res = H5Pset_char_encoding(singleton.link_property_list, H5T_CSET_UTF8);

    if (res < 0) {
        EINSUMS_LOG_ERROR("Could not apply properties to the HDF5 link property list!");
        H5Fclose(singleton.hdf5_file);
        H5Pclose(singleton.link_property_list);
        std::terminate();
    }

    res = H5Pset_create_intermediate_group(singleton.link_property_list, 1);

    if (res < 0) {
        EINSUMS_LOG_ERROR("Could not apply properties to the HDF5 link property list!");
        H5Fclose(singleton.hdf5_file);
        H5Pclose(singleton.link_property_list);
        std::terminate();
    }

    create_complex_types();
}

void initialize_Einsums_Tensor() {
    auto &singleton     = einsums::detail::Einsums_Tensor_vars::get_singleton();
    auto &global_config = GlobalConfigMap::get_singleton();

    auto fname = std::filesystem::path(global_config.get_string("scratch-dir"));
    fname /= global_config.get_string("hdf5-file-name");

    auto err = H5open();

    if (err < 0) {
        EINSUMS_LOG_ERROR("Could not initialize HDF5 library!");
        std::terminate();
    }

    // Open an existing file, but fall back to (re)creating it if the open fails -
    // a stale or corrupt einsums.<pid>.h5 left by an earlier process whose PID was
    // reused must not take the whole process down.
    if (!std::filesystem::exists(fname) || !open_hdf5_file(fname.string())) {
        create_hdf5_file(fname.string());
    }
}

void finalize_Einsums_Tensor() {
    auto &singleton     = einsums::detail::Einsums_Tensor_vars::get_singleton();
    auto &global_config = GlobalConfigMap::get_singleton();

    auto fname = std::filesystem::path(global_config.get_string("scratch-dir"));
    fname /= global_config.get_string("hdf5-file-name");

    H5Fclose(singleton.hdf5_file);

    if (singleton.hdf5_file != H5I_INVALID_HID && global_config.get_bool("delete-hdf5-files", true)) {
        H5Fdelete(fname.string().c_str(), H5P_DEFAULT); // .string(): path::c_str() is wchar_t* on Windows
    }

    if (singleton.link_property_list != H5I_INVALID_HID) {
        H5Pclose(singleton.link_property_list);
    }

    if (singleton.double_complex_type != H5I_INVALID_HID) {
        H5Tclose(singleton.double_complex_type);
    }

    if (singleton.float_complex_type != H5I_INVALID_HID) {
        H5Tclose(singleton.float_complex_type);
    }

    auto err = H5close();

    if (err < 0) {
        EINSUMS_LOG_ERROR("Error when closing the HDF5 library!");
    }
}

} // namespace einsums