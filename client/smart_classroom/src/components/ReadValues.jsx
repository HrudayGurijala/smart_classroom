import {useState,useEffect} from "react";


const ReadValues =()=>{
    const [person_count,setPersonCount] = useState(0);
    const [temperature,setTemperature] = useState(0);
    const [humidity,setHumidity] = useState(0);
    
    const fetchValues = async()=>{
        try {
            const response = await fetch("http://127.0.0.1:8000/sensor_data");
            const data = await response.json;
            setPersonCount(data.person_count);
            setTemperature(data.temperature);
            setHumidity(data.humidity);

        } catch (error) {
            console.log("Error fetching the values", error)
        }
    }

    useEffect(()=>{
        fetchValues();
    },[]);


    return (
        <>
            <p>Number of persons in the room : {person_count}</p>
            <p>Temperature of the room(in Fahrenheit) : {temperature}</p>
            <p>Humidity of the room(in %) : {humidity}</p>

            <button onClick={fetchValues}>Click for current values</button>
        </>
    )
}