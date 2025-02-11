import Left from "./Left";
import Middle from "./Middle";
import Right from "./Right";
import '../../assets/body.css'
function Body() {
  const baseAPILink = "http://127.0.0.1:5000";
  //const baseAPILink = "https://globally-loved-burro.ngrok-free.app";
  return (
    <div className="main-div-grid">
      <div className="left-div">
        <Left />
      </div>
      <div className="center-div">
        <Middle baseAPILink={baseAPILink} />
      </div>
      <div className="right-div">
        <Right />
      </div>
    </div>
  );
}

export default Body;
