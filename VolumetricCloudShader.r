#include "AEConfig.h"
#include "AE_EffectVers.h"

#ifndef AE_OS_WIN
    #include "AE_General.r"
#endif

resource 'PiPL' (16000) {
    {
        Kind {
            AEEffect
        },
        Name {
            "VolumetricCloudShader"
        },
        Category {
            "saumotion"
        },
#ifdef AE_OS_WIN
    #if defined(AE_PROC_INTELx64)
        CodeWin64X86 {"EntryPointFunc"},
    #elif defined(AE_PROC_ARM64)
        CodeWinARM64 {"EntryPointFunc"},
    #endif
#endif
        AE_PiPL_Version {
            2,
            0
        },
        AE_Effect_Spec_Version {
            PF_PLUG_IN_VERSION,
            PF_PLUG_IN_SUBVERS
        },
        AE_Effect_Version {
            524289    /* 1.0 */
        },
        AE_Effect_Info_Flags {
            0
        },
        AE_Effect_Global_OutFlags {
            0x2000414
        },
        AE_Effect_Global_OutFlags_2 {
            0x8000000
        },
        AE_Effect_Match_Name {
            "VolumetricCloudShader"
        },
        AE_Reserved_Info {
            0
        }
    }
};