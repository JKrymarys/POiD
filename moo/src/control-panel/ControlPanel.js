import React from "react";
import "./ControlPanel.css";
import { TextField } from "@material-ui/core";

function ControlPanel({ changeStartRange, changeEndRange, changeStep }) {
  return (
    <div className="control-panel">
      <TextField
        className="standard-number"
        label="Start range"
        type="number"
        onChange={({ target: { value } }) => changeStartRange(value)}
      />
      <TextField
        className="standard-number"
        label="End Range"
        type="number"
        onChange={({ target: { value } }) => changeEndRange(value)}
      />
      <TextField
        className="standard-number"
        label="Step"
        type="number"
        onChange={({ target: { value } }) => changeStep(value)}
      />
    </div>
  );
}

export default ControlPanel;
