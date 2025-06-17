document.addEventListener('DOMContentLoaded', () => {
    const colorButton = document.getElementById('colorButton');
    const body = document.body;
    
    // Array of background colors
    const colors = [
        '#f5f5f5',
        '#ffe4e1',
        '#e0ffff',
        '#f0fff0',
        '#fff0f5'
    ];
    
    let currentColorIndex = 0;
    
    // Change background color when button is clicked
    colorButton.addEventListener('click', () => {
        currentColorIndex = (currentColorIndex + 1) % colors.length;
        body.style.backgroundColor = colors[currentColorIndex];
    });
}); 