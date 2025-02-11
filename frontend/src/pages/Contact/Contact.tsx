import Topheader from "../home/TopHeader";
import "./contact.css";
const AboutPage = () => {
  return (
    <>
      <Topheader />
      <div className="main-contact-container">
        <div style={{ marginTop: "2rem", marginBottom: "2rem" }}>
          <h1>Think you got what it takes?</h1>
          <p>Reach out and help improve this website by contacting me!</p>
        </div>
        <img src="./tao_thonk.png"></img>
        <hr></hr>
        <div style={{ marginTop: "2rem" }}>
          <h2>See you there!</h2>
          <p>
            <a href="mailto:jasonavilasoria@gmail.com">@Jay</a> or{" "}
            <a href="mailto:jasonavilasoria@gmail.com">
              jasonavilasoria@gmail.com
            </a>
          </p>
        </div>
      </div>
      <Topheader />
    </>
  );
};

export default AboutPage;
