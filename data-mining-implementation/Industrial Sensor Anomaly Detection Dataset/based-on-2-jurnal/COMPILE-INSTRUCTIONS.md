# LaTeX Compilation Instructions

## File Created: `mathematical-formulas.tex`

This file contains all mathematical formulas used in the feature selection pipeline, properly formatted in LaTeX for easy reading and publication.

---

## How to Compile to PDF

### Option 1: Online (Recommended for Quick Preview)

1. **Overleaf** (easiest, no installation required):
   - Go to https://www.overleaf.com/
   - Create free account (if needed)
   - Click "New Project" → "Upload Project"
   - Upload `mathematical-formulas.tex`
   - Click "Recompile" button
   - Download PDF

2. **ShareLaTeX**:
   - Go to https://www.sharelatex.com/
   - Similar process as Overleaf

### Option 2: Local Installation (for Offline Use)

#### Windows:

1. **Install MiKTeX** (LaTeX distribution):
   ```powershell
   # Download from: https://miktex.org/download
   # Or using Chocolatey:
   choco install miktex
   ```

2. **Install TeXstudio** (LaTeX editor):
   ```powershell
   # Download from: https://www.texstudio.org/
   # Or using Chocolatey:
   choco install texstudio
   ```

3. **Compile**:
   - Open `mathematical-formulas.tex` in TeXstudio
   - Press F5 (or Tools → Build & View)
   - PDF will be generated automatically

#### Alternative: Command Line

```powershell
# Navigate to directory
cd "c:\Users\muham\OneDrive\Desktop\data-mining-uas\data-mining-implementation\Industrial Sensor Anomaly Detection Dataset\based-on-2-jurnal"

# Compile to PDF
pdflatex mathematical-formulas.tex

# If you have citations/references (run twice):
pdflatex mathematical-formulas.tex
pdflatex mathematical-formulas.tex
```

### Option 3: VS Code with LaTeX Extension

1. **Install LaTeX Workshop extension**:
   - Open VS Code
   - Go to Extensions (Ctrl+Shift+X)
   - Search "LaTeX Workshop"
   - Install by James Yu

2. **Install MiKTeX** (see Windows instructions above)

3. **Compile**:
   - Open `mathematical-formulas.tex` in VS Code
   - Press Ctrl+Alt+B (Build LaTeX)
   - Or click "Build LaTeX Project" in sidebar
   - PDF preview will open automatically

---

## What's Included in the LaTeX File

### Section 1: Introduction
- Overview of the mathematical framework

### Section 2: Filter Methods
- **Chi-Square Test**: Association measurement formula
- **Mutual Information**: Information gain calculation
- **ANOVA F-test**: Variance analysis across classes
- **Pearson Correlation**: Linear relationship measurement

### Section 3: Embedded Methods
- **Gini Importance**: Random Forest feature importance
- **LASSO**: L1 regularization objective function

### Section 4: Data Balancing
- **SMOTE**: Synthetic sample generation algorithm

### Section 5: Ensemble Voting
- **Vote Count**: Aggregation across methods
- **Threshold Selection**: Feature selection based on votes

### Section 6: Performance Metrics
- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Feature Reduction Metrics**: Reduction rate, training speedup
- **Statistical Validation**: Jaccard similarity, performance difference

### Section 7: Threshold Optimization
- **F1-Score Maximization**: Optimal threshold selection

### Section 8: Summary Table
- Quick reference for all methods and formulas

### Section 9: References
- Full citations for all referenced papers

---

## Required LaTeX Packages

The following packages are used (automatically installed by MiKTeX/TeX Live):

```latex
\usepackage{amsmath}      % Advanced math formatting
\usepackage{amssymb}      % Math symbols
\usepackage{geometry}     % Page layout
\usepackage{enumitem}     % Better lists
\usepackage{hyperref}     % Clickable references
```

---

## Output Files After Compilation

After successful compilation, you'll have:

```
mathematical-formulas.tex  (source file)
mathematical-formulas.pdf  (compiled document) ← THIS IS WHAT YOU WANT
mathematical-formulas.aux  (auxiliary file)
mathematical-formulas.log  (compilation log)
mathematical-formulas.out  (hyperref output)
```

**Important**: Keep the `.tex` file for future edits. The `.pdf` is the final readable document.

---

## Troubleshooting

### Error: "pdflatex not found"
- Make sure MiKTeX or TeX Live is installed
- Restart VS Code/terminal after installation
- Check PATH environment variable includes LaTeX binaries

### Error: "Missing package"
- MiKTeX will auto-install missing packages (click "Install" when prompted)
- Or manually: `mpm --install=<package-name>`

### Error: "Compilation failed"
- Check the `.log` file for detailed error messages
- Common issues:
  - Missing `$` for math mode
  - Unclosed braces `{}`
  - Missing `\end{document}`

### PDF not opening
- Check if PDF is locked by another program
- Close PDF reader and recompile
- Use SumatraPDF (better for live preview)

---

## Benefits of Using LaTeX for Math

✅ **Professional Typography**: Publication-quality mathematical notation  
✅ **Easy Editing**: Change formulas without worrying about alignment  
✅ **Consistency**: Uniform formatting throughout document  
✅ **Portability**: Works on any platform (Windows, Mac, Linux)  
✅ **Version Control**: Plain text format works with Git  
✅ **Academic Standard**: Required by most journals and conferences  

---

## Quick Reference: Common LaTeX Math Symbols

```latex
% Greek letters
\alpha, \beta, \gamma, \delta, \sigma, \mu, \lambda

% Operators
\sum (summation), \prod (product), \int (integral)

% Fractions
\frac{numerator}{denominator}

% Subscripts and Superscripts
x_i (subscript), x^2 (superscript), x_i^2 (both)

% Set notation
\in (element of), \cup (union), \cap (intersection)

% Brackets
\left( ... \right) (auto-sizing parentheses)
\left[ ... \right] (auto-sizing square brackets)

% Text in math mode
\text{your text here}
```

---

## Next Steps

1. ✅ Compile `mathematical-formulas.tex` to PDF
2. ✅ Review all formulas for accuracy
3. ✅ Use PDF for documentation or publication
4. ✅ If needed, export formulas to Word/PowerPoint (copy from PDF)

---

## Support

If you encounter issues:
1. Check LaTeX Workshop output in VS Code
2. Review the `.log` file for detailed errors
3. Search error messages on Stack Exchange (TeX.SE)
4. LaTeX documentation: https://www.latex-project.org/help/documentation/

**Created**: November 2024  
**Format**: LaTeX (pdflatex)  
**Purpose**: Mathematical documentation for feature selection pipeline
