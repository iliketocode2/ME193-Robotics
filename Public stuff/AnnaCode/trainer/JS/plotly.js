export function newPlot(divName, data, layout){
    Plotly.newPlot(document.getElementById(divName), data, layout, {responsive: true});
}

export function update(divName, data, layout, traces){
    if (traces !== undefined && traces !== null) {
        Plotly.update(document.getElementById(divName), data, layout || {}, traces);
    } else {
        Plotly.update(document.getElementById(divName), data, layout);
    }
}

// Restyle only the given trace indices, leaving every other trace on the
// plot untouched -- used for the fit-plot's two independent traces
// (dataset points vs. the live run trace) so updating one never touches
// the other.
export function restyle(divName, update, traces){
    Plotly.restyle(document.getElementById(divName), update, traces);
}

// Append trace(s) to an existing plot -- used to add a new Data/Live line
// pair whenever an output is added to the network.
export function addTraces(divName, traces){
    Plotly.addTraces(document.getElementById(divName), traces);
}

// Remove trace(s) by index -- used when an output is deleted. Removing
// shifts every later trace's index down, which callers must account for.
export function deleteTraces(divName, indices){
    Plotly.deleteTraces(document.getElementById(divName), indices);
}
