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

// Implement TShader methods (Remove visibility attribute from definitions)
// Add visibility attribute specifically for the constructor
__attribute__((visibility("default")))
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

const TIntermediate* TShader::getIntermediate() const {
    static TIntermediate dummyIntermediate;
    return &dummyIntermediate;
}

void TShader::setStrings(const char* const* strings, int n) {}

void TShader::setAutoMapLocations(bool map) {}

void TShader::setAutoMapBindings(bool map) {}

// Define InitializeProcess/FinalizeProcess within the namespace (Remove visibility attribute)
void InitializeProcess() {}

void FinalizeProcess() {}

} // namespace glslang

// Define global C-style functions within extern "C"
extern "C" {
    // Static global TBuiltInResource for simplicity
    TBuiltInResource DefaultTBuiltInResource;

    // Remove visibility attribute
    TBuiltInResource* GetDefaultResources() {
        DefaultTBuiltInResource.maxLights = 32;
        DefaultTBuiltInResource.maxClipPlanes = 6;
        DefaultTBuiltInResource.maxTextureUnits = 32;
        return &DefaultTBuiltInResource;
    }

    // Remove visibility attribute
    glslang::TPoolAllocator* GetThreadPoolAllocator() { 
        static glslang::TPoolAllocator pool;
        return &pool; 
    }

    // Remove visibility attribute
    int GetKhronosToolId() { return 0; }

} // extern "C"

// Define GlslangToSpv functions as regular C++ functions (Remove visibility attribute)
void glslang::GlslangToSpv(const glslang::TIntermediate& intermediate,
                 std::vector<unsigned int>& spirv,
                 glslang::SpvBuildLogger* logger,
                 glslang::SpvOptions* options) {
    spirv.clear(); // Minimal stub implementation
}

void glslang::GlslangToSpv(const glslang::TIntermediate& intermediate,
                 std::vector<unsigned int>& spirv) {
    glslang::GlslangToSpv(intermediate, spirv, nullptr, nullptr);
}

// Define the missing 3-argument overload (Remove visibility attribute)
void glslang::GlslangToSpv(const glslang::TIntermediate& intermediate,
                 std::vector<unsigned int>& spirv,
                 glslang::SpvOptions* options) {
    glslang::GlslangToSpv(intermediate, spirv, nullptr, options); // Call the 4-arg version
}
