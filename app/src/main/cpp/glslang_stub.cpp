#include <string>
#include <vector>
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

class TIntermediate {
public:
    TIntermediate() {}
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
TShader::TShader(EShLanguage stage) {}
TShader::~TShader() {}
void TShader::setStringsWithLengths(const char* const* strings, const int* lengths, int n) {}
void TShader::addProcesses(const std::vector<std::string>& processes) {}
void TShader::setEntryPoint(const char* entryPoint) {}
void TShader::setSourceEntryPoint(const char* sourceEntryPointName) {}
bool TShader::parse(const TBuiltInResource* res, int defaultVersion, EProfile profile, 
                  bool force, bool verbose, EShMessages messages) { return true; }
bool TShader::parse(const TBuiltInResource* res, int defaultVersion, EProfile profile, 
                  bool force, bool verbose, EShMessages messages, Includer& includer) { return true; }
const char* TShader::getInfoLog() { return ""; }
const char* TShader::getInfoDebugLog() { return ""; }
const TIntermediate* TShader::getIntermediate() const { return nullptr; }
void TShader::setStrings(const char* const* strings, int n) {}
void TShader::setAutoMapLocations(bool map) {}
void TShader::setAutoMapBindings(bool map) {}

// Static global TBuiltInResource for simplicity
TBuiltInResource DefaultTBuiltInResource;

// Define all necessary global functions - make sure they're exported
TPoolAllocator* GetThreadPoolAllocator() { 
    static TPoolAllocator pool;
    return &pool; 
}

int GetKhronosToolId() { return 0; }

void InitializeProcess() {}

void FinalizeProcess() {}

// Export key functions NCNN needs with C linkage to avoid name mangling issues
extern "C" {
    TBuiltInResource* GetDefaultResources() {
        return &DefaultTBuiltInResource;
    }
}

// Export GlslangToSpv with all variations needed

// Version with SpvBuildLogger and SpvOptions (matches header default)
void GlslangToSpv(const TIntermediate& intermediate,
                 std::vector<unsigned int>& spirv,
                 SpvBuildLogger* logger,
                 SpvOptions* options) {
    spirv.clear();
}

// Version with only SpvOptions* (matches linker error)
void GlslangToSpv(const TIntermediate& intermediate,
                 std::vector<unsigned int>& spirv,
                 SpvOptions* options) {
    spirv.clear();
}

// Version with no logger or options (matches header overload)
void GlslangToSpv(const TIntermediate& intermediate,
                 std::vector<unsigned int>& spirv) {
    spirv.clear();
}

} // namespace glslang
