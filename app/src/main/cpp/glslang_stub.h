#ifndef GLSLANG_STUB_H
#define GLSLANG_STUB_H

#include <string>
#include <vector>

// Forward declarations of glslang types
struct TBuiltInResource;

namespace glslang {
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
    
    class TVector;
    class TType;
    class TQualifier;
    class TIntermConstantUnion;
    class TIntermTyped;
    class TIntermediate;
    class TPoolAllocator;
    class SpvOptions;
    class SpvBuildLogger;
    class TShader;
    
    // Forward declare the GlslangToSpv function
    void GlslangToSpv(const TIntermediate& intermediate, 
                     std::vector<unsigned int>& spirv,
                     SpvBuildLogger* logger = nullptr,
                     SpvOptions* options = nullptr);
}

#endif // GLSLANG_STUB_H
