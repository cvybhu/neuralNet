CXX = g++-8
SRCS_DIR = src

DEFINES = -DCL_USE_DEPRECATED_OPENCL_1_2_APIS
CXXFLAGS = -std=c++17 -Wall -Wshadow $(DEFINES)
DEBUG_FLAGS = -DDEBUG -O0 -g -fsanitize=undefined -fsanitize=address
RELEASE_FLAGS = -DNDEBUG -O3 -flto

INCLUDE_DIRS = -I./src -I./libs
LD_DIRS = -L./libs

LD_FLAGS = $(LD_DIRS) -lstdc++fs -lOpenCL

BUILD_DIR = build
DEBUG_BUILD_DIR = $(BUILD_DIR)/debug
RELEASE_BUILD_DIR = $(BUILD_DIR)/release

SRCS = $(shell find $(SRCS_DIR) -name '*.cpp' -or -name '*.c')
DEBUG_OBJS = $(SRCS:%=$(DEBUG_BUILD_DIR)/%.o)
RELEASE_OBJS = $(SRCS:%=$(RELEASE_BUILD_DIR)/%.o)

-include $(shell find $(BUILD_DIR) -name '*.d')

$(DEBUG_BUILD_DIR)/%.cpp.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) -c $< -o $@ -MMD $(CXXFLAGS) $(INCLUDE_DIRS) $(LD_FLAGS) $(DEBUG_FLAGS)

$(RELEASE_BUILD_DIR)/%.cpp.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) -c $< -o $@ -MMD $(CXXFLAGS) $(INCLUDE_DIRS) $(LD_FLAGS) $(RELEASE_FLAGS)

debug: $(DEBUG_OBJS)
	$(CXX) $(DEBUG_OBJS) -o main $(CXXFLAGS) $(LD_FLAGS) $(DEBUG_FLAGS)

release: $(RELEASE_OBJS)
	$(CXX) $(RELEASE_OBJS) -o main $(CXXFLAGS) $(LD_FLAGS) $(RELEASE_FLAGS)

clean:
	find $(BUILD_DIR) -name "*.o" -delete && find $(BUILD_DIR) -name "*.d" -delete
