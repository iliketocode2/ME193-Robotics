export function newPlot(divName, data, layout){
    Plotly.newPlot(document.getElementById(divName), data, layout, {responsive: true});
}

export function update(divName, data, layout){
    Plotly.update(document.getElementById(divName),  data, layout);
}
