#!/usr/bin/env bash
set -e
echo "========================================================"
echo "Running ASD Multimodal Verification Test Suites"
echo "========================================================"
echo "[1/2] Running tests/test_audit_requirements.py ..."
python tests/test_audit_requirements.py
echo ""
echo "[2/2] Running tests/test_evidence_gaps.py ..."
python tests/test_evidence_gaps.py
echo ""
echo "========================================================"
echo "[SUCCESS] All verification test suites passed 100%!"
echo "========================================================"

