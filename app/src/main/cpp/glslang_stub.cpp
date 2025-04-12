#include <string>
#include <vector>

// Structure to match glslang's TBuiltInResource
struct TBuiltInResource {};

namespace glslang {

// Enums needed by the NCNN library
enum EShLanguage { EShLangVertex, EShLangFragment, EShLangCompute };
enum EProfile { ENoProfile, ECoreProfile, ECompatibilityProfile, EEsProfile };
enum EShMessages { 
    EShMsgDefault = 0,
    EShMsgRelaxedErrors = (1 << 0),
    EShMsgSuppressWarnings = (1 << 1),
    EShMsgVulkanRules = (1 << 2),
    EShMsgSpvRules = (1 << 3),
    EShMsgReadHlsl = (1 << 4) 
};

// Forward declare the std namespace classes used in function signatures
namespace std {
namespace __ndk1 {
    // These are just forward declarations - don't redefine the actual types
    template<class T> class allocator;
    template<class T, class A> class vector;
    template<class charT, class traits, class Allocator> class basic_string;
    typedef basic_string<char, std::char_traits<char>, allocator<char>> string;
}}

// Simplified class stubs - only what NCNN needs
class TVector {
public:
    TVector() {}
};

class TType {
public:
    TType() {}
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
};

// Pool allocator for the compiler
class TPoolAllocator {
public:
    void push() {}
    void pop() {}
    void* allocate(unsigned long size) { return nullptr; }
};

class SpvOptions {};

class SpvBuildLogger {};

class TShader {
public:
    // Define Includer class inside TShader
    class Includer {
    public:
        virtual ~Includer() {}
    };

    // IMPORTANT: Match exact signature from error message
    TShader(EShLanguage stage) {}
    ~TShader() {}

    // Exact method signatures from error message
    void setStringsWithLengths(const char* const* strings, const int* lengths, int n) {}
    
    void addProcesses(const std::__ndk1::vector<std::__ndk1::basic_string<char, std::__ndk1::char_traits<char>, 
                    std::__ndk1::allocator<char>>, std::__ndk1::allocator<std::__ndk1::basic_string<char, 
                    std::__ndk1::char_traits<char>, std::__ndk1::allocator<char>>>>& processes) {}
    
    void setEntryPoint(const char* entryPoint) {}
    void setSourceEntryPoint(const char* sourceEntryPointName) {}
    
    bool parse(const TBuiltInResource* res, int defaultVersion, EProfile profile, 
              bool force, bool verbose, EShMessages messages, TShader::Includer& includer) { return true; }
    
    const char* getInfoLog() { return ""; }
    const char* getInfoDebugLog() { return ""; }
};

// IMPORTANT: Define these functions directly in the glslang namespace
TPoolAllocator* GetThreadPoolAllocator() { 
    static TPoolAllocator pool;
    return &pool; 
}

int GetKhronosToolId() { return 0; }

void InitializeProcess() {}

void FinalizeProcess() {}

// Export GlslangToSpv with exact signature from error message
void GlslangToSpv(const TIntermediate& intermediate, 
                std::__ndk1::vector<unsigned int, std::__ndk1::allocator<unsigned int>>& spirv,
                SpvOptions* options = nullptr) {}

// Additional overload matching what the error messages show
void GlslangToSpv(const TIntermediate& intermediate, 
                std::__ndk1::vector<unsigned int, std::__ndk1::allocator<unsigned int>>& spirv,
                SpvBuildLogger* logger,
                SpvOptions* options) {}

} // namespace glslang
