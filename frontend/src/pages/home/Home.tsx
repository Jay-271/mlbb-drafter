import Body from "./Body";
import TopHeader from "./TopHeader";
import Alert from "../../components/Alert";
import "../../assets/home.css";
import { useState } from "react";

function App() {
  const [alertVisible, setAlertVisibility] = useState(true);

  return (
    <div className="main-div-3">
      <div className="top">
        <TopHeader />
        {alertVisible && (
          <Alert onClose={() => setAlertVisibility(false)}>
            Note | This is still in alpha phase!!
          </Alert>
        )}
      </div>

      <div className="center">
        <Body />
      </div>

      <div className="bottom">
        <TopHeader />
      </div>
    </div>
  );
}

export default App;
