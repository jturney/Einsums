# * note that upstream uses `write_basic_package_version_file(... COMPATIBILITY
#   ExactVersion)` so no version range of compatibilty can be expressed here.

include(FetchContent)

fetchcontent_declare(
    range-v3
    URL https://github.com/ericniebler/range-v3/archive/0.12.0.tar.gz
    URL_HASH SHA256=015adb2300a98edfceaf0725beec3337f542af4915cec4d0b89fa0886f4ba9cb
    FIND_PACKAGE_ARGS 0.12.0
)

fetchcontent_makeavailable(range-v3)
