#ifndef GLSLANG_STUB_H
#define GLSLANG_STUB_H

#include <string>
#include <vector>

// Forward declarations of glslang types
struct TBuiltInResource;

namespace glslang {
    // Define enumerations needed by NCNN
    enum EShLanguage { 
        EShLangVertex, 
        EShLangFragment,
        EShLangCompute
    };
    
    enum EProfile { 
        ENoProfile, 
        ECoreProfile, 
        ECompatibilityProfile, 
        EEsProfile 
    };
    
    enum EShMessages { 
        EShMsgDefault = 0,
        EShMsgRelaxedErrors = (1 << 0),
        EShMsgSuppressWarnings = (1 << 1),
        EShMsgVulkanRules = (1 << 2),
        EShMsgSpvRules = (1 << 3),
        EShMsgReadHlsl = (1 << 4) 
    };
    
    // Forward declarations
    class TVector;
    class TType;
    class TQualifier;
    class TIntermConstantUnion;
    class TIntermTyped;
    class TIntermediate;
    class TPoolAllocator;
    class SpvOptions;
    class SpvBuildLogger;
    class TString;
    
    // TShader class declaration
    class TShader {
    public:
        class Includer {
        public:
            virtual ~Includer() {}
        };
        
        TShader(EShLanguage stage);
        ~TShader();
        
        void setStringsWithLengths(const char* const* strings, const int* lengths, int n);
        void addProcesses(const std::vector<std::string>& processes);
        void setEntryPoint(const char* entryPoint);
        void setSourceEntryPoint(const char* sourceEntryPointName);
        bool parse(const TBuiltInResource* res, int defaultVersion, EProfile profile, 
                  bool force, bool verbose, EShMessages messages);
        bool parse(const TBuiltInResource* res, int defaultVersion, EProfile profile, 
                  bool force, bool verbose, EShMessages messages, Includer& includer);
        const char* getInfoLog();
        const char* getInfoDebugLog();
        const TIntermediate* getIntermediate() const;
        void setStrings(const char* const* strings, int n);
        void setAutoMapLocations(bool map);
        void setAutoMapBindings(bool map);
    };
    
    // Declare extern functions that NCNN needs
    TPoolAllocator* GetThreadPoolAllocator();
    int GetKhronosToolId();
    void InitializeProcess();
    void FinalizeProcess();
    
    // Forward declare the GlslangToSpv function
    void GlslangToSpv(const TIntermediate& intermediate, 
                     std::vector<unsigned int>& spirv,
                     SpvBuildLogger* logger = nullptr,
                     SpvOptions* options = nullptr);
    
    void GlslangToSpv(const TIntermediate& intermediate, 
                     std::vector<unsigned int>& spirv);
}

// C-style API for getting default resources (used by NCNN)
extern "C" {
    TBuiltInResource* GetDefaultResources();
}

#endif // GLSLANG_STUB_H
