#ifndef GLSLANG_STUB_H
#define GLSLANG_STUB_H

#include <string>
#include <vector>

// Forward declarations of glslang types
struct TBuiltInResource;

namespace glslang {
    // Define enumerations needed by NCNN - Match ncnn's header definition
    typedef enum {
        EShLangVertex,
        EShLangTessControl,
        EShLangTessEvaluation,
        EShLangGeometry,
        EShLangFragment,
        EShLangCompute,
        EShLangRayGen,
        EShLangIntersect,
        EShLangAnyHit,
        EShLangClosestHit,
        EShLangMiss,
        EShLangCallable,
        EShLangTask,
        EShLangMesh,
        EShLangCount, // Add count marker if present in original
    } EShLanguage;
    
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

    // Declare GlslangToSpv within the glslang namespace
    void GlslangToSpv(const glslang::TIntermediate& intermediate,
                     std::vector<unsigned int>& spirv,
                     glslang::SpvBuildLogger* logger = nullptr,
                     glslang::SpvOptions* options = nullptr);

    void GlslangToSpv(const glslang::TIntermediate& intermediate,
                     std::vector<unsigned int>& spirv);

    // Add the 3-argument overload declaration if needed by ncnn
    void GlslangToSpv(const glslang::TIntermediate& intermediate,
                     std::vector<unsigned int>& spirv,
                     glslang::SpvOptions* options);

    // Move Initialize/FinalizeProcess into the namespace
    void InitializeProcess();
    void FinalizeProcess();

} // namespace glslang

// C-style API declarations (Keep only truly C-style functions here)
extern "C" {
    TBuiltInResource* GetDefaultResources();
    glslang::TPoolAllocator* GetThreadPoolAllocator();
    int GetKhronosToolId();
}

#endif // GLSLANG_STUB_H
