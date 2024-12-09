export type ClassroomConditions = {
  temperature: number;
  humidity: number;
  peopleCount: number;
};

export type AcAdjustment = {
  suggestion: string;
  temperatureDiff: number;
  humidityDiff: number;
  energyImpact: number;
};

export function calculateAcAdjustment({
  temperature,
  humidity,
  peopleCount,
}: ClassroomConditions): AcAdjustment {
  const idealTemp = 22; // 22°C or about 72°F
  const idealHumidity = 50; // 50%
  const peopleImpact = peopleCount * 0.1; // Each person raises temp by 0.1°C

  const tempDiff = temperature - idealTemp + peopleImpact;
  const humidityDiff = humidity - idealHumidity;

  let suggestion: string;
  let energyImpact: number;

  if (tempDiff > 2 || humidityDiff > 10) {
    suggestion = "Increase cooling";
    energyImpact = 1;
  } else if (tempDiff < -2 || humidityDiff < -10) {
    suggestion = "Decrease cooling";
    energyImpact = -1;
  } else {
    suggestion = "Maintain current settings";
    energyImpact = 0;
  }

  return {
    suggestion,
    temperatureDiff: tempDiff,
    humidityDiff,
    energyImpact,
  };
}
