@echo off
title ASD Multimodal Verification Tests
echo ========================================================
echo Running ASD Multimodal Verification Test Suites
echo ========================================================
echo [1/2] Running tests\test_audit_requirements.py ...
python tests\test_audit_requirements.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] test_audit_requirements.py failed!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [2/2] Running tests\test_evidence_gaps.py ...
python tests\test_evidence_gaps.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] test_evidence_gaps.py failed!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ========================================================
echo [SUCCESS] All verification test suites passed 100%!
echo ========================================================
pause

