import Topheader from "../home/TopHeader";
import "./sources.css";
const Sources = () => {
  return (
    <>
      <Topheader />
      <div className="main-sources-container">
        <div style={{ marginTop: "2rem", marginBottom: "2rem" }}>
          <h1>Sources</h1>
          <p>
            All data for models and hero list were procured from publicaly
            available wikis like{" "}
            <a href="https://liquipedia.net/mobilelegends/Portal:Heroes">
              Mobile Legends Liquepedia
            </a>
            . For more information contact &nbsp;
            <a href="mailto:mlbbdrafter@gmail.com">@Jay</a>.
            <br />
            <strong>
              Note* All images were generated with Whisk AI by Google or a
              third-party AI-generator software. If you think this is a mistake,
              please contact me. Background image of "Lunox" belongs to Mobile
              Legends: Bang Bang.
            </strong>
          </p>
        </div>
        <img src="./tao_dawn.png"></img>
      </div>
      <Topheader />
    </>
  );
};
export default Sources;
