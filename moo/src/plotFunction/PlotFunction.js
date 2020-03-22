import React from "react";
import Chart from "chart.js";

class PlotFunction extends React.Component {
  chartRef = React.createRef();

  componentDidMount() {
    const myChartRef = this.chartRef.current.getContext("2d");
    this.myChart = new Chart(myChartRef, {
      type: "line",
      data: {
        //Bring in data
        labels: this.props.data.x,
        datasets: [
          {
            label: this.props.label,
            data: this.props.data.y,
            fill: false,
            backgroundColor: this.props.color,
            borderColor: this.props.color
          }
        ]
      },
      options: {
        // fill: false
      }
    });
  }

  render() {
    return <canvas ref={this.chartRef} />;
  }
}

export default PlotFunction;
