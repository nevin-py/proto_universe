#!/bin/bash
# Test Byzantine detection across all architectures (separate tests)

echo "=============================================================================="
echo "BYZANTINE DETECTION - All Architectures (Separate Tests)"
echo "=============================================================================="
echo ""

PASS_COUNT=0
FAIL_COUNT=0

# Test Linear Model
echo "### Testing Linear Model (10×784) ###"
if python scripts/test_byzantine_multimodel.py --model linear 2>&1 | grep -q "linear - Detection: 100%"; then
    echo "✅ Linear: PASS"
    ((PASS_COUNT++))
else
    echo "❌ Linear: FAIL"
    ((FAIL_COUNT++))
fi
echo ""

# Test MLP Model
echo "### Testing MLP Model (10×64) ###"
if python scripts/test_byzantine_mlp.py; then
    echo "✅ MLP: PASS"
    ((PASS_COUNT++))
else
    echo "❌ MLP: FAIL"
    ((FAIL_COUNT++))
fi
echo ""

# Test CNN Model
echo "### Testing CNN Model (10×128) ###"
if python scripts/test_byzantine_cnn.py; then
    echo "✅ CNN: PASS"
    ((PASS_COUNT++))
else
    echo "❌ CNN: FAIL"
    ((FAIL_COUNT++))
fi
echo ""

# Summary
echo "=============================================================================="
echo "FINAL SUMMARY"
echo "=============================================================================="
echo "Passed: $PASS_COUNT/3"
echo "Failed: $FAIL_COUNT/3"
echo ""

if [ $PASS_COUNT -eq 3 ]; then
    echo "🎉 ALL ARCHITECTURES PASSED - 100% Byzantine detection!"
    exit 0
else
    echo "⚠️  Some architectures failed"
    exit 1
fi
