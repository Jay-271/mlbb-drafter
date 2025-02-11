interface Props {
  btnName: string;
  color?: 'primary' | 'secondary' | 'danger';
  onClick: () => void;
}
function Button({ btnName, onClick, color = 'primary' }: Props) {
  return (
    <>
      <button type="button" className={"btn btn-" + color} onClick={onClick}>
        {btnName}
      </button>
    </>
  );
}
export default Button;
