LOCAL_PATH := $(call my-dir)

# Clear local variables
include $(CLEAR_VARS)

# Define local module name
LOCAL_MODULE := glslang_stub
LOCAL_SRC_FILES := glslang_stub.cpp
LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)
include $(BUILD_SHARED_LIBRARY)

# NCNN library
include $(CLEAR_VARS)
LOCAL_MODULE := ncnn
LOCAL_SRC_FILES := $(LOCAL_PATH)/libs/$(TARGET_ARCH_ABI)/libncnn.a
LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)/ncnn_include $(LOCAL_PATH)/ncnn_include/ncnn
include $(PREBUILT_STATIC_LIBRARY)

# Main native library
include $(CLEAR_VARS)
LOCAL_MODULE := native-lib
LOCAL_SRC_FILES := native-lib.cpp
LOCAL_SHARED_LIBRARIES := glslang_stub
LOCAL_STATIC_LIBRARIES := ncnn
LOCAL_LDLIBS := -llog -landroid -ljnigraphics -lvulkan
LOCAL_CFLAGS := -fopenmp -DNCNN_VULKAN=1
LOCAL_CPP_FEATURES := rtti exceptions
include $(BUILD_SHARED_LIBRARY)
