window.addEventListener('load', function () {

    // Map notebook filenames (or partial URL) to an array of zero-based cell indexes to highlight
    const highlightMap = {
        'Tutorial_00.ipynb': [1],
        'Tutorial_01.ipynb': [1, 4, 6, 10],
        'Tutorial_02.ipynb': [2, 3, 4, 5, 6, 7, 15, 19, 31],
        'Tutorial_03.ipynb': [2, 5, 8, 11, 14, 18, 21, 24, 26, 30, 34],
        'Tutorial_04.ipynb': [2, 3, 4, 6, 13, 14, 15, 19],
        'Tutorial_annotation.ipynb': [3, 6, 9, 11],
        'Tutorial_ligand_receptor.ipynb': [2, 4, 6, 8, 10, 13, 15],
        'Tutorial_marker.ipynb': [2, 4, 6, 8, 17, 21],
        'Tutorial_proportion.ipynb': [2, 4],
        'Tutorial_mullti.ipynb': [2, 5, 8, 17, 21, 23, 25, 27, 29]
        // add other notebooks here
    };

    // Get current notebook filename from location.href or metadata
    // Here we assume the URL ends with .../notebook_name.html (converted)
    const url = window.location.pathname;
    const notebookName = url.split('/').pop().replace('.html', '.ipynb');

    const indexesToColor = highlightMap[notebookName];
    if (!indexesToColor){
        return;
    } 

    function applyHighlights() {

        // ── Use the outer container instead ───────────────────────────────────
        const all_cells = document.querySelectorAll('div.nbinput.docutils');
        console.log(`Found ${all_cells.length} cells`);

        // Safety check
        const maxIndex = Math.max(...indexesToColor);
        if (all_cells.length < maxIndex + 1) {
            console.warn(`Expected ${maxIndex + 1} cells, found ${all_cells.length}`);
            return false;
        }

        indexesToColor.forEach(i => {
            const cell = all_cells[i];

            // ── Only select the code area, NOT the prompt ─────────────────────
            const code_area = cell.querySelector('.input_area.highlight-ipython3');
            if (code_area) {
                const highlights = code_area.querySelectorAll('.highlight');
                highlights.forEach(h => h.style.background = "powderblue");
            } else {
                console.warn(`No input_area found in cell ${i}`);
            }
        });

        return true;
    }

    // Try immediately
    if (applyHighlights()) return;
});
