// Copyright the Hastur authors.
// SPDX-License-Identifier: Apache-2.0
//
// Windows-only DLL loader glue. The plugin bundles onnxruntime.dll (and its CUDA
// provider DLLs) next to the .ofx in Contents/Win64. When an OFX host LoadLibrary's
// the .ofx, the loader does NOT search the plugin's own directory for that plugin's
// dependencies, so a normally-linked onnxruntime.dll would fail to resolve.
//
// We solve this WITHOUT touching the process-wide DLL search order (which could
// break the host): onnxruntime.dll is delay-loaded (see CMake /DELAYLOAD), and the
// delay-load helper calls the hook below the first time an ORT symbol is used. The
// hook loads our ONNX Runtime by explicit full path from this module's directory.
// ORT then loads its provider DLLs from that same directory itself.
//
// CRITICAL: the bundled runtime is renamed to a PRIVATE base name (onnxruntime_hastur.dll,
// see CMake OFX_LIB_BUNDLED) and the hook loads THAT, not "onnxruntime.dll". Windows
// ships its own onnxruntime.dll in System32 (Windows ML), and hosts such as Nuke make
// it resident; because the loader keys modules by base name, a LoadLibraryExW of our
// full path to a file named "onnxruntime.dll" would just return the already-resident
// System32 module (a different, ABI-incompatible ORT) — binding us to ORT 1.17 and
// failing our API-27 calls. A unique base name can never be deduplicated against it.
#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <delayimp.h>
#include <string.h>
#include <wchar.h>

static HMODULE g_selfModule = nullptr;

static FARPROC WINAPI DelayHook(unsigned event, PDelayLoadInfo info) {
  if (event == dliNotePreLoadLibrary && info != nullptr && info->szDll != nullptr &&
      _stricmp(info->szDll, "onnxruntime.dll") == 0 && g_selfModule != nullptr) {
    wchar_t path[MAX_PATH];
    DWORD n = GetModuleFileNameW(g_selfModule, path, MAX_PATH);
    if (n > 0 && n < MAX_PATH) {
      // Trim the filename, keeping the trailing separator.
      DWORD k = n;
      while (k > 0 && path[k - 1] != L'\\' && path[k - 1] != L'/') --k;
      path[k] = 0;
      // Load our PRIVATELY-NAMED runtime (never the base name "onnxruntime.dll",
      // which collides with Windows ML's System32 copy — see file header).
      static const wchar_t kDll[] = L"onnxruntime_hastur.dll";
      if (k + (DWORD)wcslen(kDll) < MAX_PATH) {
        wcscat_s(path, MAX_PATH, kDll);
        HMODULE h = LoadLibraryExW(path, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (h != nullptr) return reinterpret_cast<FARPROC>(h);
      }
    }
  }
  return nullptr;  // fall back to the default delay-load behaviour
}

// The delay-load runtime calls this hook. C linkage is declared in <delayimp.h>.
const PfnDliHook __pfnDliNotifyHook2 = DelayHook;

BOOL APIENTRY DllMain(HMODULE module, DWORD reason, LPVOID /*reserved*/) {
  if (reason == DLL_PROCESS_ATTACH) {
    g_selfModule = module;
    DisableThreadLibraryCalls(module);
  }
  return TRUE;
}

#endif  // _WIN32
