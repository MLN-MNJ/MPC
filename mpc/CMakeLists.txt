cmake_minimum_required(VERSION 3.5)
project(mpc)

# Default to C++17 (required for structured bindings)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
foreach(PACKAGE 
  ament_cmake 
  ament_cmake_python
  rclcpp
  rclpy 
  geometry_msgs
  nav_msgs
  ackermann_msgs
  sensor_msgs
  std_msgs
  visualization_msgs
  Eigen3
  tf2
  tf2_ros)

  find_package(${PACKAGE} REQUIRED)

endforeach()

find_package(OsqpEigen REQUIRED)

# Include directories
include_directories(include)
include_directories(${EIGEN3_INCLUDE_DIRS})

# ─────────────────────────────────────────────────────────────────────────────
# EXISTING: OSQP-based MPC node (lab submission)
# ─────────────────────────────────────────────────────────────────────────────
add_executable(mpc_node src/mpc_node.cpp)

target_include_directories(mpc_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIRS}
)

ament_target_dependencies(mpc_node
  rclcpp geometry_msgs ackermann_msgs nav_msgs sensor_msgs std_msgs visualization_msgs tf2 tf2_ros
)

target_link_libraries(mpc_node OsqpEigen::OsqpEigen)

install(TARGETS mpc_node
  DESTINATION lib/${PROJECT_NAME})

# ─────────────────────────────────────────────────────────────────────────────
# NEW: acados-based Racing MPC node
# Only built if c_generated_code/ exists (run generate_racing_solver.py first)
# ─────────────────────────────────────────────────────────────────────────────
set(ACADOS_SOURCE_DIR "$ENV{HOME}/acados")
set(ACADOS_INCLUDE_DIR "${ACADOS_SOURCE_DIR}/include")
set(ACADOS_LIB_DIR "${ACADOS_SOURCE_DIR}/lib")
set(MPC_GENERATED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/c_generated_code")

if(EXISTS "${MPC_GENERATED_DIR}/acados_solver_f1tenth_racing.h")
  message(STATUS "acados generated code found — building racing_mpc_node")

  file(GLOB_RECURSE ACADOS_GENERATED_SOURCES "${MPC_GENERATED_DIR}/*.c")
  
  add_library(acados_racing_solver STATIC ${ACADOS_GENERATED_SOURCES})
  target_include_directories(acados_racing_solver PUBLIC
    ${ACADOS_INCLUDE_DIR}
    ${ACADOS_INCLUDE_DIR}/blasfeo/include
    ${ACADOS_INCLUDE_DIR}/hpipm/include
    ${MPC_GENERATED_DIR}
  )
  target_link_libraries(acados_racing_solver
    ${ACADOS_LIB_DIR}/libacados.so
    ${ACADOS_LIB_DIR}/libblasfeo.so
    ${ACADOS_LIB_DIR}/libhpipm.so
  )

  add_executable(racing_mpc_node src/racing_mpc_node.cpp)

  target_include_directories(racing_mpc_node PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${EIGEN3_INCLUDE_DIRS}
    ${ACADOS_INCLUDE_DIR}
    ${ACADOS_INCLUDE_DIR}/blasfeo/include
    ${ACADOS_INCLUDE_DIR}/hpipm/include
    ${MPC_GENERATED_DIR}
  )

  ament_target_dependencies(racing_mpc_node
    rclcpp geometry_msgs ackermann_msgs nav_msgs sensor_msgs std_msgs visualization_msgs tf2 tf2_ros
  )

  target_link_libraries(racing_mpc_node
    acados_racing_solver
    ${ACADOS_LIB_DIR}/libacados.so
    ${ACADOS_LIB_DIR}/libblasfeo.so
    ${ACADOS_LIB_DIR}/libhpipm.so
  )

  install(TARGETS racing_mpc_node
    DESTINATION lib/${PROJECT_NAME})

else()
  message(STATUS "acados generated code NOT found — skipping racing_mpc_node")
  message(STATUS "Run: python3 generate_racing_solver.py  to generate it")
endif()

# ─────────────────────────────────────────────────────────────────────────────
# Python
# ─────────────────────────────────────────────────────────────────────────────
ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  scripts/mpc_node.py
  scripts/utils.py
  DESTINATION lib/${PROJECT_NAME}
)

install(FILES
  scripts/generate_racing_solver.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()