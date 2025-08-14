document.addEventListener('DOMContentLoaded', function () {
    // Map notebook filenames (or partial URL) to an array of zero-based cell indexes to highlight
    const highlightMap = {
        'Tutorial_00.ipynb': [1],
        'Tutorial_01.ipynb': [1, 4, 6, 10],
        'Tutorial_02.ipynb': [2, 3, 4, 5, 6, 7, 15, 19, 31],
        'Tutorial_03.ipynb': [2, 5, 8, 11, 14, 18, 21, 24, 26, 30, 34],
        'Tutorial_04.ipynb': [2, 3, 4, 6, 13, 14, 15, 19]
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

    // Select code cells in the page — adjust selector if needed for your output format
    let all_cells = document.querySelectorAll('.input_area')
    let selection = indexesToColor

    let sel_cells = selection.map(x=>all_cells[x]);
    let ca = sel_cells.map(e=>Array.from(e.querySelectorAll('.highlight'))).flat();
    ca.forEach(e=>e.style.background="powderblue")
});