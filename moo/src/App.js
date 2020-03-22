import React from "react";
import PlotFunction from "./plotFunction/PlotFunction";
import range from "lodash/range";
import "./App.css";

function App() {
  //mock data
  const startRange = -10;
  const endRange = 10;
  const step = 1;
  const ax = range(startRange, endRange, step);
  console.log("ax", ax);
  const fx = ax.map(x => Math.pow(x, 2));
  const plotData = {
    x: ax,
    y: fx
  };

  return (
    <div className="App">
      <PlotFunction data={plotData} label={"y = x^2"} color="#ff0000" />
    </div>
  );
}

export default App;
