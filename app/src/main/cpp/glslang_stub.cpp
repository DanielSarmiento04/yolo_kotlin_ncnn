#include <string>
#include <vector>

// Need to match the expected signatures exactly
namespace std {
namespace __ndk1 {
    template<class T> class allocator {};
    template<class T, class Alloc = allocator<T>> class vector {
    public:
        vector() {}
        void push_back(const T&) {}
        size_t size() const { return 0; }
        T* data() { return nullptr; }
    };
    
    template<class charT, class traits = char_traits<charT>, class Allocator = allocator<charT>>
    class basic_string {
    public:
        basic_string() {}
        basic_string(const charT*) {}
    };
    
    typedef basic_string<char> string;
    template<class K, class V> struct pair { K first; V second; };
}}

// Define missing glslang symbols with full signatures
struct TBuiltInResource {};

namespace glslang {

enum EShLanguage { EShLangVertex, EShLangFragment };
enum EProfile { ENoProfile };
enum EShMessages { EShMsgDefault };

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
    
    // Match all required functions
    TShader(EShLanguage stage) {}
    ~TShader() {}
    void setStringsWithLengths(const char* const* strings, const int* lengths, int n) {}
    void addProcesses(const std::__ndk1::vector<std::__ndk1::string>& processes) {}
    void setEntryPoint(const char* entryPoint) {}
    void setSourceEntryPoint(const char* sourceEntryPointName) {}
    bool parse(const TBuiltInResource* res, int defaultVersion, EProfile profile, 
              bool force, bool verbose, EShMessages messages, TShader::Includer& includer) { return true; }
    const char* getInfoLog() { return ""; }
    const char* getInfoDebugLog() { return ""; }
};

// Stub implementations for all missing functions with full exact signatures
extern "C" {
    TPoolAllocator* GetThreadPoolAllocator() { 
        static TPoolAllocator pool;
        return &pool; 
    }
    
    int GetKhronosToolId() { return 0; }
    
    void InitializeProcess() {}
    
    void FinalizeProcess() {}
}

// Make sure this signature exactly matches what libncnn.a expects
void GlslangToSpv(const TIntermediate& intermediate, 
                 std::__ndk1::vector<unsigned int>& spirv,
                 SpvBuildLogger* logger = nullptr,
                 SpvOptions* options = nullptr) {}
} // namespace glslang

// Do not define any symbols in ncnn namespace - they're already in libncnn.a
