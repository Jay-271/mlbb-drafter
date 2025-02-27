import Topheader from "../home/TopHeader";
import "./about.css";
const AboutPage = () => {
  return (
    <>
      <Topheader />
      <div className="main-about-container">
        <div style={{ marginTop: "2rem", marginBottom: "2rem" }}>
          <h1>Glad to see you're interested in my app!</h1>
          <p>Hopefully this AI drafter has helped :)</p>
        </div>

        <div style={{ marginBottom: "1rem", padding: "1rem" }}>
          <h2>Meet the Team</h2>
          <ul style={{ listStyleType: "none", padding: 0 }}>
            <li>🚀 Jay - Lead Developer</li>
            <li>🛰️ Jay - UX Designer</li>
            <li>☄️ Jay - Project Manager</li>
          </ul>
        </div>
        <img src="./tao_think.png"></img>
        <hr></hr>
        <div style={{marginTop: "2rem"}}>
          <h2>Contact Us</h2>
          ...
          <p>
            Have questions? Reach out to &nbsp;
            <a href="mailto:mlbbdrafter@gmail.com">@Jay</a>.
          </p>
...
        </div>
        
      </div>
      <Topheader />

    </>
  );
};

export default AboutPage;
