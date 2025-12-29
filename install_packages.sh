#!/bin/bash
# Quick package installation script

echo "Installing required packages..."
echo ""

# Try to install system-wide (requires --break-system-packages on modern Linux)
pip3 install --break-system-packages --upgrade pip
pip3 install --break-system-packages -r requirements.txt

echo ""
echo "✓ Installation complete!"
echo ""
echo "To verify installation, run:"
echo "  python3 -c \"import pandas, numpy, sklearn, matplotlib, seaborn, imblearn; print('All packages installed!')\""
echo ""
echo "Note: If you're using Jupyter, you may need to restart the kernel after installation."


