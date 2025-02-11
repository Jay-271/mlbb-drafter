import { Link } from "react-router-dom";

interface Props {
  middleText: string;
  Links: string[];
}
const Header = ({ middleText, Links }: Props) => {
  return (
    <div className="header">
      <div className="logo">
        <img className="logo-img" src="./drafter_logo.jpg"></img>
      </div>
      <div className="header-subtitle">
        <span className="text">
          <strong>{middleText}</strong>
        </span>
      </div>
      <div className="header-navigation">
        <nav className="nav">
          <ul className="nav-list">
            {Links.map((item) => (
              <li key={String(item)}>
                <Link to={item.startsWith("/") ? item : `/${item}`}>
                  {item}
                </Link>
              </li>
            ))}
          </ul>
        </nav>
      </div>
    </div>
  );
};

export default Header;
