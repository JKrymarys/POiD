import React from "react";
import Chart from "chart.js";

import "./ChartPanel.css";

class ChartPanel extends React.Component {
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
        maintainAspectRatio: false
      }
    });
  }

  componentDidUpdate() {
    const { x: labels, y: data, color } = this.props.data;

    console.log("X", labels);
    console.log("fx", data);

    this.myChart.data.labels = labels;
    this.myChart.data.datasets = [
      {
        label: "test",
        data: data,
        fill: false,
        backgroundColor: color,
        borderColor: color
      }
    ];
    this.myChart.update();

    console.log("CHART UPDATED");
  }

  render() {
    return (
      <div className="chart-container">
        <canvas ref={this.chartRef} />
      </div>
    );
  }
}

export default ChartPanel;
