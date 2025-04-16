#include <string>
#include <vector>
#include <map> // Include the map header
#include "glslang_stub.h"
#include <android/log.h> // For logging
#include <cstdlib> // For malloc/free

#define LOG_TAG_STUB "glslang_stub"
#define LOGI_STUB(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG_STUB, __VA_ARGS__)
#define LOGE_STUB(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG_STUB, __VA_ARGS__)

// Structure to match glslang's TBuiltInResource - Expanded with common fields
struct TBuiltInResource {
    int maxLights;
    int maxClipPlanes;
    int maxTextureUnits;
    int maxTextureCoords;
    int maxVertexAttribs;
    int maxVertexUniformComponents;
    int maxVaryingFloats; // Deprecated, use maxVaryingComponents
    int maxVertexOutputComponents;
    int maxGeometryInputComponents;
    int maxGeometryOutputComponents;
    int maxFragmentInputComponents;
    int maxImageUnits;
    int maxCombinedTextureImageUnits;
    int maxCombinedShaderOutputResources;
    int maxUniformBufferBindings;
    int maxAtomicCounterBindings;
    int maxVertexAtomicCounters;
    int maxTessControlAtomicCounters;
    int maxTessEvaluationAtomicCounters;
    int maxGeometryAtomicCounters;
    int maxFragmentAtomicCounters;
    int maxCombinedAtomicCounters;
    int maxAtomicCounterBuffers;
    int maxVertexAtomicCounterBuffers;
    int maxTessControlAtomicCounterBuffers;
    int maxTessEvaluationAtomicCounterBuffers;
    int maxGeometryAtomicCounterBuffers;
    int maxFragmentAtomicCounterBuffers;
    int maxCombinedAtomicCounterBuffers;
    int maxAtomicCounterBufferSize;
    int maxTransformFeedbackBuffers;
    int maxTransformFeedbackInterleavedComponents;
    int maxCullDistances;
    int maxCombinedClipAndCullDistances;
    int maxSamples;
    int maxMeshOutputVerticesNV;
    int maxMeshOutputPrimitivesNV;
    int maxMeshWorkGroupSizeX_NV;
    int maxMeshWorkGroupSizeY_NV;
    int maxMeshWorkGroupSizeZ_NV;
    int maxTaskWorkGroupSizeX_NV;
    int maxTaskWorkGroupSizeY_NV;
    int maxTaskWorkGroupSizeZ_NV;
    int maxMeshViewCountNV;
    int maxDualSourceDrawBuffersEXT;

    struct Limits {
        bool nonInductiveForLoops;
        bool whileLoops;
        bool doWhileLoops;
        bool generalUniformIndexing;
        bool generalAttributeMatrixVectorIndexing;
        bool generalVaryingIndexing;
        bool generalSamplerIndexing;
        bool generalVariableIndexing;
        bool generalConstantMatrixVectorIndexing;
    } limits;
};

namespace glslang {

// Forward declarations for missing types used internally
class TIntermNode {}; // Minimal definition
class SpvVersion {};  // Minimal definition
class TSymbol {};     // Minimal definition

// Define string class implementation if not fully included
class TString {
    std::string s_;
public:
    TString(const char* s = "") : s_(s ? s : "") {}
    TString(const std::string& str) : s_(str) {}
    const char* c_str() const { return s_.c_str(); }
    // Add other methods if needed by the stub implementations
};

// Implement forward declared classes minimally
class TType {};
class TVector {};
class TQualifier {};
class TIntermConstantUnion {};
class TIntermTyped {};

// Minimal TIntermediate implementation
class TIntermediate {
public:
    TIntermediate() { LOGI_STUB("TIntermediate constructor"); }
    ~TIntermediate() { LOGI_STUB("TIntermediate destructor"); }
    const TType* getOutputType() const { return nullptr; }
    bool postProcess(TIntermNode*, EShLanguage) { return true; }
    void setEntryPointName(const char*) {}
    void setEntryPointMangledName(const char*) {}
    void setVersion(int) {}
    void setSpv(const SpvVersion&) {}
    void addRequestedExtension(const char*) {}
    const std::map<std::string, TSymbol*>& getSymbolTable() const { static std::map<std::string, TSymbol*> map; return map; }
    // Add other minimal methods if linker errors point to them
};

class TPoolAllocator {
public:
    void push() {}
    void pop() {}
    // A simple allocation strategy for the stub (not production quality)
    void* allocate(unsigned long size) { return malloc(size); }
    void deallocate(void* ptr) { free(ptr); } // Need a way to deallocate if used
};

class SpvOptions {};
class SpvBuildLogger {};

// Implement TShader methods

TShader::TShader(EShLanguage stage) : lang(stage), intermediate(nullptr) {
    LOGI_STUB("TShader constructor called for stage: %d", stage);
    intermediate = new TIntermediate();
}

TShader::~TShader() {
    LOGI_STUB("TShader destructor called");
    delete intermediate;
}

void TShader::setStringsWithLengths(const char* const* strings, const int* lengths, int n) {
    LOGI_STUB("TShader::setStringsWithLengths called");
}

void TShader::addProcesses(const std::vector<std::string>& processes) {
    LOGI_STUB("TShader::addProcesses called");
}

void TShader::setEntryPoint(const char* entryPoint) {
    LOGI_STUB("TShader::setEntryPoint called: %s", entryPoint ? entryPoint : "nullptr");
}

void TShader::setSourceEntryPoint(const char* sourceEntryPointName) {
    LOGI_STUB("TShader::setSourceEntryPoint called: %s", sourceEntryPointName ? sourceEntryPointName : "nullptr");
}

bool TShader::parse(const TBuiltInResource* res, int defaultVersion, EProfile profile,
                  bool force, bool verbose, EShMessages messages) {
    LOGI_STUB("TShader::parse (no Includer) called");
    infoLog = "Parse successful (stub)";
    return true; // Indicate success
}

bool TShader::parse(const TBuiltInResource* res, int defaultVersion, EProfile profile,
                  bool force, bool verbose, EShMessages messages, Includer& includer) {
    LOGI_STUB("TShader::parse (with Includer) called");
    infoLog = "Parse successful (stub with Includer)";
    return true; // Indicate success
}

const char* TShader::getInfoLog() {
    LOGI_STUB("TShader::getInfoLog called");
    return infoLog.c_str();
}

const char* TShader::getInfoDebugLog() {
    LOGI_STUB("TShader::getInfoDebugLog called");
    return debugLog.c_str();
}

const TIntermediate* TShader::getIntermediate() const {
    LOGI_STUB("TShader::getIntermediate called");
    return intermediate;
}

void TShader::setStrings(const char* const* strings, int n) {
     LOGI_STUB("TShader::setStrings called");
}

void TShader::setAutoMapLocations(bool map) {
    LOGI_STUB("TShader::setAutoMapLocations called: %d", map);
}

void TShader::setAutoMapBindings(bool map) {
    LOGI_STUB("TShader::setAutoMapBindings called: %d", map);
}

void InitializeProcess() {
    LOGI_STUB("glslang::InitializeProcess called");
}

void FinalizeProcess() {
    LOGI_STUB("glslang::FinalizeProcess called");
}

void GlslangToSpv(const glslang::TIntermediate& intermediate,
                 std::vector<unsigned int>& spirv,
                 glslang::SpvBuildLogger* logger,
                 glslang::SpvOptions* options) {
    LOGI_STUB("glslang::GlslangToSpv (4 args) called");
    spirv.clear();
}

void GlslangToSpv(const glslang::TIntermediate& intermediate,
                 std::vector<unsigned int>& spirv) {
    LOGI_STUB("glslang::GlslangToSpv (2 args) called");
    GlslangToSpv(intermediate, spirv, nullptr, nullptr);
}

void GlslangToSpv(const glslang::TIntermediate& intermediate,
                 std::vector<unsigned int>& spirv,
                 glslang::SpvOptions* options) {
    LOGI_STUB("glslang::GlslangToSpv (3 args) called");
    GlslangToSpv(intermediate, spirv, nullptr, options);
}

} // namespace glslang

// Define global C-style functions within extern "C"
extern "C" {
    // Static global TBuiltInResource for simplicity
    static TBuiltInResource DefaultTBuiltInResource = {
        /* .maxLights = */ 32,
        /* .maxClipPlanes = */ 6,
        /* .maxTextureUnits = */ 32,
        /* .maxTextureCoords = */ 32,
        /* .maxVertexAttribs = */ 64,
        /* .maxVertexUniformComponents = */ 4096,
        /* .maxVaryingFloats = */ 64,
        /* .maxVertexOutputComponents = */ 64,
        /* .maxGeometryInputComponents = */ 64,
        /* .maxGeometryOutputComponents = */ 128,
        /* .maxFragmentInputComponents = */ 128,
        /* .maxImageUnits = */ 32,
        /* .maxCombinedTextureImageUnits = */ 80,
        /* .maxCombinedShaderOutputResources = */ 32,
        /* .maxUniformBufferBindings = */ 32, // Added
        /* .maxAtomicCounterBindings = */ 1, // Added
        /* .maxVertexAtomicCounters = */ 0, // Added
        /* .maxTessControlAtomicCounters = */ 0, // Added
        /* .maxTessEvaluationAtomicCounters = */ 0, // Added
        /* .maxGeometryAtomicCounters = */ 0, // Added
        /* .maxFragmentAtomicCounters = */ 8, // Added
        /* .maxCombinedAtomicCounters = */ 8, // Added
        /* .maxAtomicCounterBuffers = */ 1, // Added
        /* .maxVertexAtomicCounterBuffers = */ 0, // Added
        /* .maxTessControlAtomicCounterBuffers = */ 0, // Added
        /* .maxTessEvaluationAtomicCounterBuffers = */ 0, // Added
        /* .maxGeometryAtomicCounterBuffers = */ 0, // Added
        /* .maxFragmentAtomicCounterBuffers = */ 1, // Added
        /* .maxCombinedAtomicCounterBuffers = */ 1, // Added
        /* .maxAtomicCounterBufferSize = */ 16384, // Added
        /* .maxTransformFeedbackBuffers = */ 4, // Added
        /* .maxTransformFeedbackInterleavedComponents = */ 64, // Added
        /* .maxCullDistances = */ 8, // Added
        /* .maxCombinedClipAndCullDistances = */ 8, // Added
        /* .maxSamples = */ 4, // Added
        /* .maxMeshOutputVerticesNV = */ 256, // Added (NV specific, might not be needed)
        /* .maxMeshOutputPrimitivesNV = */ 512, // Added (NV specific)
        /* .maxMeshWorkGroupSizeX_NV = */ 32, // Added (NV specific)
        /* .maxMeshWorkGroupSizeY_NV = */ 1, // Added (NV specific)
        /* .maxMeshWorkGroupSizeZ_NV = */ 1, // Added (NV specific)
        /* .maxTaskWorkGroupSizeX_NV = */ 32, // Added (NV specific)
        /* .maxTaskWorkGroupSizeY_NV = */ 1, // Added (NV specific)
        /* .maxTaskWorkGroupSizeZ_NV = */ 1, // Added (NV specific)
        /* .maxMeshViewCountNV = */ 4, // Added (NV specific)
        /* .maxDualSourceDrawBuffersEXT = */ 1, // Added (EXT specific)
        /* .limits = */ {
            /* .nonInductiveForLoops = */ true,
            /* .whileLoops = */ true,
            /* .doWhileLoops = */ true,
            /* .generalUniformIndexing = */ true,
            /* .generalAttributeMatrixVectorIndexing = */ true,
            /* .generalVaryingIndexing = */ true,
            /* .generalSamplerIndexing = */ true,
            /* .generalVariableIndexing = */ true,
            /* .generalConstantMatrixVectorIndexing = */ true
        }
    };

    TBuiltInResource* GetDefaultResources() {
        LOGI_STUB("GetDefaultResources called");
        return &DefaultTBuiltInResource;
    }

    glslang::TPoolAllocator* GetThreadPoolAllocator() {
        LOGI_STUB("GetThreadPoolAllocator called");
        static glslang::TPoolAllocator pool;
        return &pool;
    }

    int GetKhronosToolId() {
        LOGI_STUB("GetKhronosToolId called");
        return 0;
    }

} // extern "C"
