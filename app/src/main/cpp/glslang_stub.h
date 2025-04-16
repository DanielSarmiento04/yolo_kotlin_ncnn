#ifndef GLSLANG_STUB_H
#define GLSLANG_STUB_H

#include <string>
#include <vector>
#include <map> // Include map for TIntermediate stub

// Forward declarations of glslang types
struct TBuiltInResource; // Keep as struct forward declaration

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
        // Add other flags if needed based on ncnn usage
    };

    // Forward declarations for types used in TShader and GlslangToSpv
    class TVector;
    class TType;
    class TQualifier;
    class TIntermConstantUnion;
    class TIntermTyped;
    class TIntermediate;
    class TPoolAllocator;
    class SpvOptions;
    class SpvBuildLogger;
    class TString; // Add forward declaration for TString

    // TShader class declaration
    class TShader {
    public:
        // Minimal Includer definition (required by one of the parse overloads)
        class Includer {
        public:
            struct IncludeResult {
                IncludeResult(const std::string& headerName, const char* const headerData, const size_t headerLength, void* userData)
                    : headerName(headerName), headerData(headerData), headerLength(headerLength), userData(userData) {}
                const std::string headerName;
                const char* const headerData;
                const size_t headerLength;
                void* userData;

            private: // Disallow copying and assignment
                IncludeResult(const IncludeResult&);
                IncludeResult& operator=(const IncludeResult&);
            };

            // NCNN might call includeLocal or includeSystem
            virtual IncludeResult* includeLocal(const char* /*headerName*/, const char* /*includerName*/, size_t /*inclusionDepth*/) { return nullptr; }
            virtual IncludeResult* includeSystem(const char* /*headerName*/, const char* /*includerName*/, size_t /*inclusionDepth*/) { return nullptr; }
            virtual void releaseInclude(IncludeResult* result) { if (result) delete result; } // Basic cleanup
            virtual ~Includer() = default;
        };

        // Constructor declaration
        TShader(EShLanguage stage);
        ~TShader();

        void setStringsWithLengths(const char* const* strings, const int* lengths, int n);
        void addProcesses(const std::vector<std::string>& processes);
        void setEntryPoint(const char* entryPoint);
        void setSourceEntryPoint(const char* sourceEntryPointName);

        // Declaration for the parse overload WITHOUT Includer
        bool parse(const TBuiltInResource* res, int defaultVersion, EProfile profile,
                  bool force, bool verbose, EShMessages messages);

        // Declaration for the parse overload WITH Includer& (This is the one causing the linker error)
        bool parse(const TBuiltInResource* res, int defaultVersion, EProfile profile,
                  bool force, bool verbose, EShMessages messages, Includer& includer);

        const char* getInfoLog();
        const char* getInfoDebugLog();
        const TIntermediate* getIntermediate() const;
        void setStrings(const char* const* strings, int n);
        void setAutoMapLocations(bool map);
        void setAutoMapBindings(bool map);

    private:
        // Add minimal internal state if needed by the stub implementations
        EShLanguage lang;
        std::string infoLog;
        std::string debugLog;
        TIntermediate* intermediate; // Pointer to manage lifetime
    };

    // Declare GlslangToSpv within the glslang namespace
    void GlslangToSpv(const glslang::TIntermediate& intermediate,
                     std::vector<unsigned int>& spirv,
                     glslang::SpvBuildLogger* logger = nullptr,
                     glslang::SpvOptions* options = nullptr);

    // Keep the 2-argument overload declaration
    void GlslangToSpv(const glslang::TIntermediate& intermediate,
                     std::vector<unsigned int>& spirv);

    // Add the 3-argument overload declaration
    void GlslangToSpv(const glslang::TIntermediate& intermediate,
                     std::vector<unsigned int>& spirv,
                     glslang::SpvOptions* options);

    // Move Initialize/FinalizeProcess into the namespace
    void InitializeProcess();
    void FinalizeProcess();

} // namespace glslang

// C-style API declarations (Keep only truly C-style functions here)
extern "C" {
    // Ensure TBuiltInResource is defined before being used here
    struct TBuiltInResource; // Forward declaration is sufficient if definition is in .cpp
    TBuiltInResource* GetDefaultResources();
    glslang::TPoolAllocator* GetThreadPoolAllocator();
    int GetKhronosToolId();
}

#endif // GLSLANG_STUB_H
