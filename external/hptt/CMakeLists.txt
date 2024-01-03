# * Not using hptt is just asking for slow sorts.
# * find_package() is disabled since we need patched source
# * hptt is built as an object library and not installed

include(FetchContent)

fetchcontent_declare(
    hptt
    URL https://github.com/Einsums/hptt/archive/v1.0.5+25.tar.gz
    URL_HASH SHA256=a93f3de8bc187e93a49c46d4a2f4a79541a5f2fb84532ed93654ab9d8f490712
    SYSTEM OVERRIDE_FIND_PACKAGE
)

fetchcontent_makeavailable(hptt)
