import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import {
  calculateAcAdjustment,
  ClassroomConditions,
} from "../utils/calculateAcAdjustment";
import { Thermometer, Droplets, Users } from "lucide-react";

interface ClassroomCardProps extends ClassroomConditions {
  id: number;
  name: string;
}

export function ClassroomCard({
  name,
  temperature,
  humidity,
  peopleCount,
}: ClassroomCardProps) {
  const { suggestion } = calculateAcAdjustment({
    temperature,
    humidity,
    peopleCount,
  });

  return (
    <Card>
      <CardHeader>
        <CardTitle>{name}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Thermometer className="h-4 w-4 text-red-500" />
              <span>Temperature</span>
            </div>
            <span>{temperature}°C</span>
          </div>
          <Progress value={(temperature / 40) * 100} className="h-2" />

          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Droplets className="h-4 w-4 text-blue-500" />
              <span>Humidity</span>
            </div>
            <span>{humidity}%</span>
          </div>
          <Progress value={humidity} className="h-2" />

          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Users className="h-4 w-4 text-green-500" />
              <span>Occupancy</span>
            </div>
            <span>{peopleCount}</span>
          </div>
          <Progress value={(peopleCount / 30) * 100} className="h-2" />

          <div className="mt-4">
            <p className="font-semibold">AC Adjustment:</p>
            <p
              className={`text-lg ${suggestion === "Increase cooling" ? "text-red-500" : suggestion === "Decrease cooling" ? "text-blue-500" : "text-green-500"}`}
            >
              {suggestion}
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
