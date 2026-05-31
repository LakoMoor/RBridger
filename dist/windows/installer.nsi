; NSIS installer for Rusty Bridge
; Build with: makensis installer.nsi
; (Run from dist/windows/ after cargo build --release --target x86_64-pc-windows-msvc -p rusty-bridge-ui)

!define APP_NAME    "Rusty Bridge"
!define APP_EXE     "rusty-bridge-ui.exe"
!define APP_ID      "RustyBridge"
!define VERSION     "0.1.0"
!define PUBLISHER   "ovROG"
!define INSTALL_DIR "$PROGRAMFILES64\${APP_NAME}"

Name "${APP_NAME} ${VERSION}"
OutFile "..\..\dist\out\RustyBridger-${VERSION}-windows-setup.exe"
InstallDir "${INSTALL_DIR}"
InstallDirRegKey HKLM "Software\${APP_ID}" "InstallDir"
RequestExecutionLevel admin
SetCompressor /SOLID lzma

!include "MUI2.nsh"
!define MUI_ABORTWARNING
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "Russian"
!insertmacro MUI_LANGUAGE "English"

Section "Main" SecMain
  SetOutPath "$INSTDIR"
  File "..\..\target\release\${APP_EXE}"
  File /nonfatal "..\..\ui\resources\rb.ico"

  WriteRegStr HKLM "Software\${APP_ID}" "InstallDir" "$INSTDIR"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_ID}" \
    "DisplayName" "${APP_NAME}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_ID}" \
    "DisplayIcon" "$INSTDIR\rb.ico"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_ID}" \
    "UninstallString" "$INSTDIR\uninstall.exe"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_ID}" \
    "DisplayVersion" "${VERSION}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_ID}" \
    "Publisher" "${PUBLISHER}"
  WriteUninstaller "$INSTDIR\uninstall.exe"

  CreateDirectory "$SMPROGRAMS\${APP_NAME}"
  CreateShortcut "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk" "$INSTDIR\${APP_EXE}" "" "$INSTDIR\rb.ico"
  CreateShortcut "$SMPROGRAMS\${APP_NAME}\Uninstall.lnk"   "$INSTDIR\uninstall.exe"
  CreateShortcut "$DESKTOP\${APP_NAME}.lnk"                "$INSTDIR\${APP_EXE}" "" "$INSTDIR\rb.ico"
SectionEnd

Section "Uninstall"
  Delete "$INSTDIR\${APP_EXE}"
  Delete "$INSTDIR\rb.ico"
  Delete "$INSTDIR\uninstall.exe"
  RMDir  "$INSTDIR"
  Delete "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk"
  Delete "$SMPROGRAMS\${APP_NAME}\Uninstall.lnk"
  RMDir  "$SMPROGRAMS\${APP_NAME}"
  Delete "$DESKTOP\${APP_NAME}.lnk"
  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_ID}"
  DeleteRegKey HKLM "Software\${APP_ID}"
SectionEnd
