import { ClassroomCard } from "./components/classroom-card";
import { Analytics } from "./components/analytics";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useEffect, useState } from "react";

// Define the type for the sensor data
type SensorData = {
  person_count: number;
  temperature: number;
  humidity: number;
};

// Define the type for classroom props
type Classroom = {
  id: number;
  name: string;
  temperature: number;
  humidity: number;
  peopleCount: number;
};
const mockclassrooms = [
  { id: 1, name: "Room 101", temperature: 24, humidity: 55, peopleCount: 20 },
  { id: 2, name: "Room 102", temperature: 22, humidity: 50, peopleCount: 15 },
  { id: 3, name: "Room 103", temperature: 26, humidity: 60, peopleCount: 25 },
  { id: 4, name: "Room 104", temperature: 21, humidity: 45, peopleCount: 10 },
  { id: 5, name: "Room 105", temperature: 23, humidity: 52, peopleCount: 18 },
];

export default function App() {
  const [person_count, setPersonCount] = useState<number>(0);
  const [temperature, setTemperature] = useState<number>(0);
  const [humidity, setHumidity] = useState<number>(0);

  const fetchValues = async (): Promise<void> => {
    try {
      const response = await fetch("http://127.0.0.1:8000/sensor_data");
      const data: SensorData = await response.json();
      console.log(data);
      setPersonCount(data.person_count);
      setTemperature(data.temperature);
      setHumidity(data.humidity);
    } catch (error) {
      console.error("Error fetching the values", error);
    }
  };

  const classrooms: Classroom[] = [
    {
      id: 1,
      name: "Room 1",
      temperature,
      humidity,
      peopleCount: person_count,
    },
  ];

  useEffect(() => {
    fetchValues();
  }, []);

  return (
    <div className="container mx-auto py-8">
      <h1 className="text-3xl font-bold mb-6">
        Classroom AC Adjustment System
      </h1>
      <Tabs defaultValue="classrooms">
        <TabsList className="mb-4">
          <TabsTrigger value="classrooms">Classrooms</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
        </TabsList>
        <TabsContent value="classrooms">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {mockclassrooms.map((classroom) => (
              <ClassroomCard key={classroom.id} {...classroom} />
            ))}
          </div>
        </TabsContent>
        <TabsContent value="analytics">
          <Analytics />
        </TabsContent>
      </Tabs>
    </div>
  );
}
