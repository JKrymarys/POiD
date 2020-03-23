import React, { useState, useEffect } from "react";
import ChartPanel from "./chart-panel/ChartPanel";
import range from "lodash/range";
import "./App.css";
import ControlPanel from "./control-panel/ControlPanel";

function App() {
  const [startRange, setStartRange] = useState(0);
  const [endRange, setEndRange] = useState(10);
  const [step, setStep] = useState(1);
  const [data, setData] = useState({
    x: [1, 2, 3],
    y: [1, 4, 9]
  });

  useEffect(() => {
    const ax = range(startRange, endRange, step);
    const fx = ax.map(x => Math.pow(x, 2));

    console.log("RECALCULATE");
    console.log("Current data", data);

    setData({ x: ax, y: fx });
  }, [startRange, endRange, step]);

  return (
    <div className="main-view">
      {console.log("data k mac", data)}
      <ChartPanel data={data} label={"y = x^2"} color="#ff0000" />
      <ControlPanel
        changeStartRange={setStartRange}
        changeEndRange={setEndRange}
        changeStep={setStep}
      />
    </div>
  );
}

export default App;
