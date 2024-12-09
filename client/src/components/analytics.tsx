"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  ScatterChart,
  Scatter,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

// Static data for the graphs
const peopleVsTemp = [
  { people: 5, temperature: 20 },
  { people: 10, temperature: 22 },
  { people: 15, temperature: 23 },
  { people: 20, temperature: 25 },
  { people: 25, temperature: 26 },
  { people: 30, temperature: 28 },
];

const peopleVsHumidity = [
  { people: 5, humidity: 40 },
  { people: 10, humidity: 45 },
  { people: 15, humidity: 50 },
  { people: 20, humidity: 55 },
  { people: 25, humidity: 60 },
  { people: 30, humidity: 65 },
];

const tempVsTime = [
  { day: "Mon", temperature: 22 },
  { day: "Tue", temperature: 23 },
  { day: "Wed", temperature: 24 },
  { day: "Thu", temperature: 22 },
  { day: "Fri", temperature: 23 },
  { day: "Sat", temperature: 21 },
  { day: "Sun", temperature: 20 },
];

const humidityVsTime = [
  { day: "Mon", humidity: 50 },
  { day: "Tue", humidity: 52 },
  { day: "Wed", humidity: 55 },
  { day: "Thu", humidity: 53 },
  { day: "Fri", humidity: 51 },
  { day: "Sat", humidity: 48 },
  { day: "Sun", humidity: 47 },
];

const peopleVsTime = [
  { hour: "9AM", people: 10 },
  { hour: "10AM", people: 15 },
  { hour: "11AM", people: 20 },
  { hour: "12PM", people: 25 },
  { hour: "1PM", people: 22 },
  { hour: "2PM", people: 18 },
  { hour: "3PM", people: 15 },
  { hour: "4PM", people: 12 },
];

export function Analytics() {
  return (
    <div className="space-y-8">
      <Card>
        <CardHeader>
          <CardTitle>Number of People vs Temperature</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid />
              <XAxis
                type="number"
                dataKey="people"
                name="people"
                unit=" people"
              />
              <YAxis
                type="number"
                dataKey="temperature"
                name="temperature"
                unit="°C"
              />
              <Tooltip cursor={{ strokeDasharray: "3 3" }} />
              <Scatter
                name="People vs Temperature"
                data={peopleVsTemp}
                fill="#8884d8"
              />
            </ScatterChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Number of People vs Humidity</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid />
              <XAxis
                type="number"
                dataKey="people"
                name="people"
                unit=" people"
              />
              <YAxis
                type="number"
                dataKey="humidity"
                name="humidity"
                unit="%"
              />
              <Tooltip cursor={{ strokeDasharray: "3 3" }} />
              <Scatter
                name="People vs Humidity"
                data={peopleVsHumidity}
                fill="#82ca9d"
              />
            </ScatterChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Average Temperature vs Time (Week)</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart
              data={tempVsTime}
              margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="day" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="temperature"
                stroke="#8884d8"
                name="Avg Temperature (°C)"
              />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Average Humidity vs Time (Week)</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart
              data={humidityVsTime}
              margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="day" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="humidity"
                stroke="#82ca9d"
                name="Avg Humidity (%)"
              />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Number of People vs Time (Day)</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart
              data={peopleVsTime}
              margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="hour" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="people"
                stroke="#ffc658"
                name="Number of People"
              />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  );
}
