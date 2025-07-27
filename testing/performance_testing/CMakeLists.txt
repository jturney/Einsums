set(performance_sources src/main.cpp)

find_package(benchmark REQUIRED)

add_library(einsums_performance OBJECT ${performance_sources})
target_include_directories(einsums_performance PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(einsums_performance PRIVATE einsums_private_flags )
target_link_libraries(einsums_performance PUBLIC Einsums::Einsums benchmark::benchmark)
