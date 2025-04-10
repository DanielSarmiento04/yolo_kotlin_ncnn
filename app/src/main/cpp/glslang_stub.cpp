#include <string>
#include <vector>

// Basic type definitions to match glslang expectations
namespace glslang {

class TPoolAllocator {
public:
    void push() {}
    void pop() {}
    void* allocate(unsigned long size) { return nullptr; }
};

class TVector {
};

class TType {
};

class TQualifier {
};

class TIntermConstantUnion;
class TIntermTyped;
class TIntermediate {
public:
    bool improperStraddle(const TType&, int, int) { return false; }
    int getBaseAlignmentScalar(const TType&, int&) { return 0; }
    int getMemberAlignment(const TType&, int&, int&, int, bool) { return 0; }
    TVector findLinkerObjects() const { return TVector(); }
};

enum EShLanguage { EShLangVertex, EShLangFragment };
enum EProfile { ENoProfile };
enum EShMessages { EShMsgDefault };

class SpvOptions {
};

class SpvBuildLogger {
};

class TShader {
public:
    TShader(EShLanguage) {}
    ~TShader() {}
    void setStringsWithLengths(const char* const*, const int*, int) {}
    void addProcesses(const std::vector<std::string>&) {}
    void setEntryPoint(const char*) {}
    void setSourceEntryPoint(const char*) {}
    bool parse(const void*, int, EProfile, bool, bool, EShMessages, Includer&) { return true; }
    const char* getInfoLog() { return ""; }
    const char* getInfoDebugLog() { return ""; }

    class Includer {
    public:
        virtual ~Includer() {}
    };
};

// Stub implementations for all missing functions
TPoolAllocator* GetThreadPoolAllocator() { return nullptr; }
int GetKhronosToolId() { return 0; }
void InitializeProcess() {}
void FinalizeProcess() {}

void GlslangToSpv(const TIntermediate&, std::vector<unsigned int>&, SpvBuildLogger*, SpvOptions* options = nullptr) {}

}  // namespace glslang

// NCNN uses these in gpu.cpp
namespace ncnn {
void destroy_gpu_instance() {}
void create_gpu_instance(const char*) {}
bool compile_spirv_module(const char*, int, const void*, std::vector<unsigned int>&) { return true; }
}
