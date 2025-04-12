#include <string>
#include <vector>
#include <map> // Include the map header
#include "glslang_stub.h"

// Structure to match glslang's TBuiltInResource
struct TBuiltInResource {
    // Provide minimal default values needed by NCNN
    int maxLights = 32;
    int maxClipPlanes = 6;
    int maxTextureUnits = 32;
    // Add other fields as needed by NCNN
};

namespace glslang {

// Forward declarations for missing types
class TIntermNode; // Forward declare TIntermNode
class SpvVersion;  // Forward declare SpvVersion
class TSymbol;     // Forward declare TSymbol

// Define string class implementation
class TString {
public:
    TString(const char* s = "") {}
    const char* c_str() const { return ""; }
};

// Implement forward declared classes
class TType {
public:
    TType() {}
};

class TVector {
public:
    TVector() {}
};

class TQualifier {
public:
    TQualifier() {}
};

class TIntermConstantUnion {
public:
    TIntermConstantUnion() {}
};

class TIntermTyped {
public:
    TIntermTyped() {}
};

// Minimal TIntermediate implementation
class TIntermediate {
public:
    TIntermediate() {}
    const TType* getOutputType() const { return nullptr; }
    // Use the forward-declared TIntermNode
    bool postProcess(TIntermNode*, EShLanguage) { return true; }
    void setEntryPointName(const char*) {}
    void setEntryPointMangledName(const char*) {}
    void setVersion(int) {}
    // Use the forward-declared SpvVersion
    void setSpv(const SpvVersion&) {}
    void addRequestedExtension(const char*) {}
    // Use the forward-declared TSymbol and std::map
    const std::map<std::string, TSymbol*>& getSymbolTable() const { static std::map<std::string, TSymbol*> map; return map; }
    bool improperStraddle(const TType& type, int offset, int len) const { return false; }
    int getBaseAlignmentScalar(const TType& type, int& offset) const { return 0; }
    int getMemberAlignment(const TType& type, int& offset, int& memberSize, 
                          int layoutPacking, bool roundUp) const { return 0; }
    TVector findLinkerObjects() const { return TVector(); }
    TIntermTyped* findFunction(bool ignorePrototypes) const { return nullptr; }
    bool setLocalSize(int dim, int size) { return true; }
    void addToCallGraph(const TString& caller, const TString& callee) {}
    TIntermediate* getIntermediate() const { return nullptr; }
};

class TPoolAllocator {
public:
    void push() {}
    void pop() {}
    void* allocate(unsigned long size) { return nullptr; }
};

class SpvOptions {};
class SpvBuildLogger {};

// Implement TShader methods (NOT redefining the class)
// Add visibility attribute to ensure export
__attribute__((visibility("default")))
TShader::TShader(EShLanguage stage) {}

__attribute__((visibility("default")))
TShader::~TShader() {}

__attribute__((visibility("default")))
void TShader::setStringsWithLengths(const char* const* strings, const int* lengths, int n) {}

__attribute__((visibility("default")))
void TShader::addProcesses(const std::vector<std::string>& processes) {}

__attribute__((visibility("default")))
void TShader::setEntryPoint(const char* entryPoint) {}

__attribute__((visibility("default")))
void TShader::setSourceEntryPoint(const char* sourceEntryPointName) {}

__attribute__((visibility("default")))
bool TShader::parse(const TBuiltInResource* res, int defaultVersion, EProfile profile, 
                  bool force, bool verbose, EShMessages messages) { return true; }

__attribute__((visibility("default")))
bool TShader::parse(const TBuiltInResource* res, int defaultVersion, EProfile profile, 
                  bool force, bool verbose, EShMessages messages, Includer& includer) { return true; }

__attribute__((visibility("default")))
const char* TShader::getInfoLog() { return ""; }

__attribute__((visibility("default")))
const char* TShader::getInfoDebugLog() { return ""; }

__attribute__((visibility("default")))
const TIntermediate* TShader::getIntermediate() const {
    return nullptr;
}

__attribute__((visibility("default")))
void TShader::setStrings(const char* const* strings, int n) {}

__attribute__((visibility("default")))
void TShader::setAutoMapLocations(bool map) {}

__attribute__((visibility("default")))
void TShader::setAutoMapBindings(bool map) {}

// Static global TBuiltInResource for simplicity
TBuiltInResource DefaultTBuiltInResource;

// Define all necessary global functions - make sure they're exported
// Add visibility attribute
__attribute__((visibility("default")))
TPoolAllocator* GetThreadPoolAllocator() { 
    static TPoolAllocator pool;
    return &pool; 
}

__attribute__((visibility("default")))
int GetKhronosToolId() { return 0; }

__attribute__((visibility("default")))
void InitializeProcess() {}

__attribute__((visibility("default")))
void FinalizeProcess() {}

// Export key functions NCNN needs with C linkage to avoid name mangling issues
extern "C" {
    __attribute__((visibility("default"))) // Add visibility here too
    TBuiltInResource* GetDefaultResources() {
        DefaultTBuiltInResource.maxLights = 32;
        DefaultTBuiltInResource.maxClipPlanes = 6;
        DefaultTBuiltInResource.maxTextureUnits = 32;
        return &DefaultTBuiltInResource;
    }
}

// Export GlslangToSpv with all variations needed
// Add visibility attribute
__attribute__((visibility("default")))
void GlslangToSpv(const TIntermediate& intermediate,
                 std::vector<unsigned int>& spirv,
                 SpvBuildLogger* logger,
                 SpvOptions* options) {
    spirv.clear();
}

__attribute__((visibility("default")))
void GlslangToSpv(const TIntermediate& intermediate,
                 std::vector<unsigned int>& spirv,
                 SpvOptions* options) {
    spirv.clear();
}

__attribute__((visibility("default")))
void GlslangToSpv(const TIntermediate& intermediate,
                 std::vector<unsigned int>& spirv) {
    spirv.clear();
}

} // namespace glslang
