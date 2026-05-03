

## Plan: Notebook Data Reporting & Readability

**What/Why:**  
- After data loading, print the number of samples and class counts for train/test splits (cats vs dogs).
- Refactor so each code cell is focused on a single logic step.
- Add markdown cells to clearly explain each step.

---

### Steps

1. **Data Reporting**
   - After any data loading or split, add a code cell that prints:
     - Number of samples loaded
     - Value counts for train/test, cats/dogs in each split

2. **Cell Refactoring**
   - Split any code cell that does more than one logical step (e.g., loading + splitting, splitting + model creation).
   - Ensure each cell is focused: e.g., one for loading, one for splitting, one for reporting, one for model setup, etc.

3. **Markdown Explanations**
   - Add markdown cells before each major step:
     - Data loading
     - Data splitting
     - Data statistics reporting
     - Model creation
     - Training/validation
   - Each markdown cell should briefly explain what the next code cell does and why.

4. **Repeat for all notebooks:**
   - data_and_loaders.ipynb
   - part1_solution.ipynb
   - part2_solution.ipynb
   - (Optionally: part3_solution.ipynb for consistency)

---

### Verification

- Each notebook, after data loading/splitting, prints the sample counts and class breakdowns.
- No code cell is overloaded; each is focused and small.
- Every major step is preceded by a clear markdown explanation.

---

Would you like to include part3_solution.ipynb as well, or just the first three? If you approve, I’ll proceed with the edits.