export function newPlot(divName, data, layout){
    Plotly.newPlot(document.getElementById(divName), data, layout, {responsive: true, staticPlot: true});
}

export function update(divName, data, layout){
    Plotly.update(document.getElementById(divName),  data, layout);
}

export function react(divName, data, layout, config){
    Plotly.react(document.getElementById(divName), data, layout, config);
}

export function purge(divName){
    Plotly.purge(document.getElementById(divName));
}

// Calls cb(customdata) when a point that carries customdata is clicked (Plotly's click lands on the nearest point
// within hoverdistance, i.e. the one the hover tag is pointing at). The page also tracks whether the pointer is
// over the plot, so it can hold off redrawing -- a redraw clears the hover and the click would find nothing.
export function onPointClick(divName, cb){
    const el = document.getElementById(divName);
    el.on('plotly_click', (ev) => {
        const p = ev.points && ev.points[0];
        if (p && p.customdata !== undefined) cb(p.customdata);
    });
    if (el._insideBound) return;   // the DOM listeners survive a purge; only the plotly one needs re-adding
    el._insideBound = true;
    el.addEventListener('mouseenter', () => { el._inside = true; });
    el.addEventListener('mouseleave', () => { el._inside = false; });
}

// True while the pointer is over the plot (the page holds off redrawing it so the hover tag is not wiped mid-click)
export function pointerInside(divName){
    return !!document.getElementById(divName)._inside;
}

// Calls cb(JSON string of the customdata ids) when points are box/lasso-selected
export function onSelect(divName, cb){
    document.getElementById(divName).on('plotly_selected', (ev) => {
        const ids = ((ev && ev.points) || []).map(p => p.customdata).filter(v => v !== undefined);
        if (ids.length) cb(JSON.stringify(ids));
    });
}
