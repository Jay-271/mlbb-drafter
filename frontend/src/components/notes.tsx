import { useState } from "react";
import Alert from "./Alert";
import Buttons from "./Buttons";
import ListGroup from "./ListGroup";

function App() {
  let items = ["New York", "Tokyo", "New Bell", "Tokgyatt", "New rizz"];
  const handleSelectItem = (item: string) => {
    console.log(item);
  };

  const [alertVisible, setAlertVisibility] = useState(true);

  return (
    <>
      <div>
        {alertVisible && (
          <Alert onClose={() => setAlertVisibility(false)}>
            Note | This is still in alpha phase!!
          </Alert>
        )}
        <h1>
          Welcome to the World's First
        </h1>
        <h2>Mobile Legends: Bang Bang AI-Drafter!</h2>
        <ListGroup
          items={items}
          heading="Cities"
          onSelectItem={handleSelectItem}
        />
        <ListGroup
          items={["Bozo", "Rogwater"]}
          heading="blobbery"
          onSelectItem={handleSelectItem}
        />

        <Buttons
          btnName="My button!"
          onClick={() => setAlertVisibility(true)}
          color="secondary"
        />

      </div>
    </>
  );
}
export default App;
