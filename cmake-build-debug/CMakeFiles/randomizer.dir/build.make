# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /cygdrive/c/Users/User/.CLion2019.3/system/cygwin_cmake/bin/cmake.exe

# The command to remove a file.
RM = /cygdrive/c/Users/User/.CLion2019.3/system/cygwin_cmake/bin/cmake.exe -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /cygdrive/c/myfiles/MyClionProjects/randomizer

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /cygdrive/c/myfiles/MyClionProjects/randomizer/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/randomizer.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/randomizer.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/randomizer.dir/flags.make

CMakeFiles/randomizer.dir/main.cpp.o: CMakeFiles/randomizer.dir/flags.make
CMakeFiles/randomizer.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/cygdrive/c/myfiles/MyClionProjects/randomizer/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/randomizer.dir/main.cpp.o"
	/usr/bin/c++.exe  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/randomizer.dir/main.cpp.o -c /cygdrive/c/myfiles/MyClionProjects/randomizer/main.cpp

CMakeFiles/randomizer.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/randomizer.dir/main.cpp.i"
	/usr/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /cygdrive/c/myfiles/MyClionProjects/randomizer/main.cpp > CMakeFiles/randomizer.dir/main.cpp.i

CMakeFiles/randomizer.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/randomizer.dir/main.cpp.s"
	/usr/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /cygdrive/c/myfiles/MyClionProjects/randomizer/main.cpp -o CMakeFiles/randomizer.dir/main.cpp.s

# Object files for target randomizer
randomizer_OBJECTS = \
"CMakeFiles/randomizer.dir/main.cpp.o"

# External object files for target randomizer
randomizer_EXTERNAL_OBJECTS =

randomizer.exe: CMakeFiles/randomizer.dir/main.cpp.o
randomizer.exe: CMakeFiles/randomizer.dir/build.make
randomizer.exe: CMakeFiles/randomizer.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/cygdrive/c/myfiles/MyClionProjects/randomizer/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable randomizer.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/randomizer.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/randomizer.dir/build: randomizer.exe

.PHONY : CMakeFiles/randomizer.dir/build

CMakeFiles/randomizer.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/randomizer.dir/cmake_clean.cmake
.PHONY : CMakeFiles/randomizer.dir/clean

CMakeFiles/randomizer.dir/depend:
	cd /cygdrive/c/myfiles/MyClionProjects/randomizer/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /cygdrive/c/myfiles/MyClionProjects/randomizer /cygdrive/c/myfiles/MyClionProjects/randomizer /cygdrive/c/myfiles/MyClionProjects/randomizer/cmake-build-debug /cygdrive/c/myfiles/MyClionProjects/randomizer/cmake-build-debug /cygdrive/c/myfiles/MyClionProjects/randomizer/cmake-build-debug/CMakeFiles/randomizer.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/randomizer.dir/depend

